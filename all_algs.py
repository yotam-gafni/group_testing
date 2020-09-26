import random
import numpy as np
import math
from copy import copy
import time

import operator as op
from functools import reduce

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
session = WolframLanguageSession()


MAX_ROUNDS = 200
MIN_ROUNDS = 10 
ITERATIONS = 50
N = 1000
k = 10

lamb = k/N
THRESH_VARIANCE = 0.01

random.seed(100)
np.random.seed(100)

# binomial coefficient n over r
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def empirical_mean_variance(samples):
        a = sum(samples) / len(samples)
        var_array = [(elem - a)**2 for elem in samples]
        if len(samples) > 1:
                emp_var = sum(var_array) / (len(samples) - 1)
        else:
                emp_var = None
        return a, emp_var

# checks if the relative error of our samples is small enough, can then stop sampling. 
def is_thresh_emp_var(samples):
	if len(samples) < MIN_ROUNDS:
		return False

	emp_mean, emp_var = empirical_mean_variance(samples)

	if emp_mean == 0:
		return True

	if math.sqrt(emp_var) / (math.sqrt(len(samples)) * emp_mean**2) < THRESH_VARIANCE:
		return True
	else:
		return False

# a common logic in many algorithms is to give the items scores, then choose the k highest/lowest ranking to be the suspected defectives
def top_k_from_item_scores(iscores, k = k, reverseVal =True):
	iscores.sort(key= lambda x: x[1], reverse=reverseVal)
	pred_defective = set([item[0] for item in iscores[:k]])
	pred_non_defective = set([item[0] for item in iscores[k:]])

	return pred_defective, pred_non_defective


def SEPDEC(t_res, t2i, i2t):
	iscores = []
	p = k/N
	for item in i2t.keys():
		score = 0
		for ti in i2t[item]:
			m = len(t2i[ti])
			denom = 0
			numer = 0
			for l in range(m+1):
				if t_res[ti] == 0:
					denom += ncr(m, l) * p**l * (1-p)**(m-l) * (1 - phi) * theta**l
				elif t_res[ti] == 1:
					denom += ncr(m, l) * p**l * (1-p)**(m-l) * (1 - (1 - phi) * theta**l)

			for l_pre in range(m):
				l = l_pre + 1
				if t_res[ti] == 0:
					numer += ncr(m-1, l_pre) * p**l_pre * (1-p)**(m-l_pre) * (1 - phi) * theta**l
				elif t_res[ti] == 1:
					numer += ncr(m-1, l_pre) * p**l_pre * (1-p)**(m-l_pre) * (1 - (1 - phi) * theta**l)	

			if numer == 0:
				score = -10000
				break
			else:
				score += math.log(numer/denom, 2)

		iscores.append(tuple([item, score]))

	return top_k_from_item_scores(iscores)



def NDD(t_res, t2i, i2t):

	iscores = []
	for item in i2t.keys():
		sumi = 0
		for ti in i2t[item]:
			if t_res[ti] == 0:
				sumi += 1
		iscores.append(tuple([item, sumi]))

	filt_size = 3*k

	filt1, pred_non_defective = top_k_from_item_scores(iscores, k = filt_size, reverseVal = False)

	iscores2 = []
	for item in filt1:
		sumi = 0
		for ti in i2t[item]:
			if t_res[ti] == 1 and len(t2i[ti].intersection(filt1)) == 1:
				sumi += 1
		iscores2.append(tuple([item, sumi]))


	pred_defective, extra_pnd = top_k_from_item_scores(iscores2)
	pred_non_defective = pred_non_defective.union(extra_pnd)

	return pred_defective, pred_non_defective


# Mathematica interface
def run_lp(constraints, variables, opt_goal):

    constraint_string = "{" + ",".join(constraints) + "}"

    variable_string = "{" + ",".join(variables) + "}"

    lp_string = "Quiet[Minimize[" + opt_goal + ", " + constraint_string + "," + variable_string + "]]"

    res = session.evaluate(wlexpr(lp_string))
    return res

def LP(t_res, t2i, i2t):

	scaling = 100
	variables = []
	constraints = []

	opt_goal = ""
	for item in i2t.keys():
		variables.append("z{}".format(item))
		constraints.append("z{} >= 0".format(item))
		constraints.append("z{} <= 1".format(item))
		opt_goal += "z{} + ".format(item)

	opt_goal += "{} * (".format(scaling)

	for test in t2i.keys():
		variables.append("eps{}".format(test))
		constraints.append("eps{} >= 0".format(test))
		if t_res[test] == 1:
			constraints.append("eps{} <= 1".format(test))
			cons = ""
			for it in t2i[test]:
				cons += "z{} + ".format(it)
			cons += "eps{} >= 1".format(test)
			constraints.append(cons)
		elif t_res[test] == 0:
			if len(t2i[test]) != 0:
				cons = ""
				for it in t2i[test]:
					cons += "z{} + ".format(it)
				cons += "0 == eps{}".format(test)
				constraints.append(cons)

		opt_goal += "eps{} + ".format(test)

	opt_goal += "0)"

	res = run_lp(constraints, variables, opt_goal)	

	iscores = []
	pred_defective = set([])
	pred_non_defective = set([])
	for item in i2t.keys():
		if res[1][item][1] != 0:
			val = res[1][item][1]
			if not (type(res[1][item][1]) == type(2) or type(res[1][item][1]) == type(2.0)):
				val = res[1][item][1][0] / res[1][item][1][1]
			iscores.append(tuple([item, val]))
		else:
			pred_non_defective.add(item)

	pred_defective, extra_pnd = top_k_from_item_scores(iscores)
	pred_non_defective = pred_non_defective.union(extra_pnd)
	

	return pred_defective, pred_non_defective.union(extra_pnd)



def NCOMP(t_res, t2i, i2t):
	item_scores = []
	for item in i2t.keys():
		i_tests = i2t[item]
		if len(i_tests) == 0:
			item_scores.append(tuple([item,k/N]))
			continue
		sum_pos = 0
		for test in i_tests:
			sum_pos += t_res[test]

		item_scores.append(tuple([item,sum_pos / len(i_tests)]))

	return top_k_from_item_scores(item_scores)


def BP(t_res, t2i, i2t):
	L_it = {}
	L_ti = {}
	init_val = math.log(lamb/(1 - lamb)) 
	def_val = math.log(theta)
	for i in range(N):
		L_it[i] = {}
		for test in range(TESTS):
			if i == 0:
				L_ti[test] = {}
			L_it[i][test] = init_val
	for iterate in range(ITERATIONS):
		for test in range(TESTS):
			for i in t2i[test]:
				if t_res[test] == 1: 
					neigh = copy(t2i[test])
					neigh.remove(i)
					mult = 1
					for j in neigh:
						mult *= (theta + (1-theta)/(1 + math.e**L_it[j][test]))

					L_ti[test][i] = math.log(theta + (1 - theta)/ (1 - (1-phi)* mult))

				elif t_res[test] == 0:
					L_ti[test][i] = def_val

		for i in range(N):
			for test in i2t[i]:
				neigh = copy(i2t[i])
				neigh.remove(test)
				summa = init_val
				for t in neigh:
					summa += L_ti[t][i]
				L_it[i][test] = summa

	# we save in mu_ti in reverse order, first i and then test, so it's easy to extract later
	mu_ti = {}
	P_i = []
	for test in range(TESTS):
		for i in t2i[test]:
			if i not in mu_ti:
				mu_ti[i] = {}
			exp = math.e**L_ti[test][i]
			mu_ti[i][test] = exp / (1 + exp)

	for i in range(N):
		if i in mu_ti:
			i_tests = mu_ti[i].keys()
			mult1 = 1
			mult0 = 1
			for it in i_tests:
				mult1 *= mu_ti[i][it]
				mult0 *= (1 - mu_ti[i][it])

			P_i.append(tuple([i,mult1 / (mult1 + mult0)]))
		else:
			P_i.append(tuple([i,k/N]))

	P_i.sort(key= lambda x: x[1], reverse=True)
	pred_defective = set([item[0] for item in P_i[:k]])
	pred_non_defective = set([item[0] for item in P_i[k:]])

	return pred_defective, pred_non_defective





phi = 0.01
theta = 0.05
alg_func = NCOMP

for TESTS in [100,150,200,250]:
	success_count = 0
	how_wrong_histograms = [[],[],[],[]]
	sum_how_wrong = [0,0,0,0]
	stopping_time = [MAX_ROUNDS] * 4
	start = time.time()
	for round_num in range(MAX_ROUNDS):
		print("In round {}".format(round_num))
		defective = []
		non_defective = [i for i in range(N)]
		for j in range(k):
			new_def = random.choice(non_defective)
			non_defective.remove(new_def)
			defective.append(new_def)

		non_defective = set(non_defective)
		defective = set(defective)

		# Running the baseline
		if is_thresh_emp_var(how_wrong_histograms[0]) and round_num < stopping_time[0]:
			stopping_time[0] = round_num

		if round_num < stopping_time[0]:
			results_defective = set([])
			results_non_defective = set([])
			for item in range(N):
				l = 0
				if item in defective:
					l = 1

				res = np.random.choice([1,0], p=[1 - (1-phi)*theta**l, (1-phi)*theta**l])	
				if res:
					results_defective.add(item)
				else:
					results_non_defective.add(item)

			how_wrong = len(non_defective.difference(results_non_defective)) + len(defective.difference(results_defective))

			sum_how_wrong[0] += how_wrong
			how_wrong_histograms[0].append(how_wrong)

		for test_regime in [0,1,2]:
			if is_thresh_emp_var(how_wrong_histograms[1 + test_regime]) and round_num < stopping_time[1 + test_regime]:
				stopping_time[1 + test_regime] = round_num

			if round_num < stopping_time[1 + test_regime]:
				if test_regime == 0:
					t_res = {}
					i2t = {}
					t2i = {}
					for test in range(TESTS):
						participation_prob = 1/k
						t2i[test] = set([])
						res = np.random.choice([1,0], size=N, p=[participation_prob, 1 - participation_prob])
						for item in range(N):
							if item not in i2t:
								i2t[item] = set([])
							if res[item]:
								t2i[test].add(item)
								i2t[item].add(test)
						l = len(defective.intersection(t2i[test]))
						t_res[test] = np.random.choice([1,0], p=[1 - (1-phi)*theta**l, (1-phi)*theta**l])

				elif test_regime == 1:
					t_res = {}
					i2t = {}
					t2i = {}
					for test in range(TESTS):
						t2i[test] = set(random.sample(range(N), k=int(N/k)))
						for item in range(N):
							if item not in i2t:
								i2t[item] = set([])
							if item in t2i[test]:
								i2t[item].add(test)
						l = len(defective.intersection(t2i[test]))
						t_res[test] = np.random.choice([1,0], p=[1 - (1-phi)*theta**l, (1-phi)*theta**l])

				elif test_regime == 2:
					t_res = {}
					i2t = {}
					t2i = {}
					t2i_ratio = int(N/k)
					i2t_ratio = int(t2i_ratio * TESTS / N)
					for i in range(k):
						t2i_curr_match = set([i*t2i_ratio + j for j in range(t2i_ratio)])
						i2t_curr_match = set([i*i2t_ratio + j for j in range(i2t_ratio)])
						for j in range(i2t_ratio):
							t2i[i*i2t_ratio + j] = copy(t2i_curr_match)
						for j in range(t2i_ratio):
							i2t[i*t2i_ratio + j] = copy(i2t_curr_match)


					is_perfect = True
					mixsteps = 0

					mixtime = 100*1000
					tests = [i for i in range(TESTS)]
					items = [i for i in range(N)]
					item_per_test_set = [i for i in range(t2i_ratio)]
					test_per_item_set = [i for i in range(i2t_ratio)]

					pm_item = np.random.choice(items, size = mixtime)
					item_choice = np.random.choice(item_per_test_set, size = mixtime)
					test_choice = np.random.choice(test_per_item_set, size = mixtime)
					norm = (1/(N*TESTS) + 1/N + 1/TESTS)
					npm_case = np.random.choice([0,1,2], size = mixtime, p=[(1/(N*TESTS)) / norm, (1/N) / norm, (1/TESTS) / norm])
					npm_tests = np.random.choice(tests, size=mixtime)
					npm_items = np.random.choice(items, size=mixtime)
					for i in range(mixtime):
						if is_perfect:
							item = pm_item[i]
							test = list(i2t[pm_item[i]])[test_choice[i]]
							i2t[item].remove(test)
							t2i[test].remove(item)
							is_perfect = False

						else:
							if npm_case[i] == 0:
								i2t[item].add(test)
								t2i[test].add(item)
								is_perfect = True

							elif npm_case[i] == 1:
								new_test = npm_tests[i]
								if new_test != test and new_test not in i2t[item]:
									out_item = list(t2i[new_test])[item_choice[i]]
									if not out_item in t2i[test]:
										i2t[item].add(new_test)
										t2i[new_test].remove(out_item)
										t2i[new_test].add(item)
										i2t[out_item].remove(new_test)
										item = out_item

							elif npm_case[i] == 2:
								new_item = npm_items[i]

								if new_item != item and not new_item in t2i[test]:
									out_test = list(i2t[new_item])[test_choice[i]]
									if not out_test in i2t[item]:
										t2i[test].add(new_item)
										i2t[new_item].remove(out_test)
										i2t[new_item].add(test)
										t2i[out_test].remove(new_item)
										test = out_test


					print("Finished mixing")
					for test in range(TESTS):
						l = len(defective.intersection(t2i[test]))
						t_res[test] = np.random.choice([1,0], p=[1 - (1-phi)*theta**l, (1-phi)*theta**l])

				alg_num = 0
				pred_defective, pred_non_defective = alg_func(t_res, t2i, i2t)
				how_wrong = len(non_defective.difference(pred_non_defective)) + len(defective.difference(pred_defective))
				sum_how_wrong[1 + test_regime] += how_wrong
				how_wrong_histograms[1+test_regime].append(how_wrong)

	for i in range(4):
		print("TESTS: {}. Alg func: {}. Avg wrong {}, stopping time: {}".format(TESTS, alg_func, sum_how_wrong[i] / stopping_time[i], stopping_time[i]))
		print("TESTS: {}. Alg func: {}. Full histogram {}".format(TESTS, alg_func, how_wrong_histograms[i]))

end = time.time()

print(end - start)
