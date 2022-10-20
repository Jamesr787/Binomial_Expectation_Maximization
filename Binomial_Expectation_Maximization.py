"""
Author: James Roberts
Last Update: 10/20/2022

This program uses dynamic programming, expectation maximization,
and Bayes' rule to determine if each set of tosses are
from coin A or B and determine their parameters
(probability of landing heads)

Bayes' Rule:
P(Class given Data) = P(Data given Class) * P(Class) / P(Data)

where

P(Data) = P(Data given Coin A) * P(Coin A) + P(Data given Coin B) * P(Coin B)
"""

import math

# Data
DATA = [[9, 1],     # Event 1 [heads, tails]
        [6, 4],     # Event 2
        [3, 7],     # Event 3
        [7, 3],     # Event 4
        [3, 7],     # Event 5
        [5, 5],     # Event 6
        [7, 3],     # Event 7
        [5, 5]]     # Event 8

# Assume priors are equal P(Class)
p_class = 0.5

# Initial Parameters (Probability of heads)
p_heads_a = 0.6       # coin A
p_heads_b = 0.4       # coin B


def binomial_probability(total, successes, probability_of_success):
        """
        This function returns the binomial probability.
        """
        combination = math.factorial(total)/ \
                      (math.factorial(total-successes)*math.factorial(successes))

        return combination * \
               (probability_of_success)**successes * \
               (1-probability_of_success)**(total-successes)




def parameter_estimation(parameters, data, prior):
        """
        This function takes parameters, data, and priors as an argument
        and returns the final parameters once EM converges. 
        """
        p_a = parameters[0]
        p_b = parameters[1]

        p_data_given_a = []
        p_data_given_b = []
        p_data = []
        p_a_given_data = []
        p_b_given_data = []

        coinA_heads_tails = []
        coinB_heads_tails = []


        for event in range(len(data)):

                # find likelihood (Probability of data given class)
                p_data_given_a.append(binomial_probability(
                        total = sum(data[event]),
                        successes = data[event][0],
                        probability_of_success = p_a))

                p_data_given_b.append(binomial_probability(
                        total=sum(data[event]),
                        successes=data[event][0],
                        probability_of_success=p_b))

                # find evidence (Probability of Data)
                p_data.append(p_data_given_a[event] * prior + p_data_given_b[event] * prior)

                # find posteriors (Probability of coin given data)
                p_a_given_data.append(p_data_given_a[event] * prior / p_data[event])
                p_b_given_data.append(p_data_given_b[event] * prior / p_data[event])

                # heads and tails for each coin
                coinA_heads_tails.append([p_a_given_data[event] * data[event][0],
                                          p_a_given_data[event] * data[event][1]])

                coinB_heads_tails.append([p_b_given_data[event] * data[event][0],
                                          p_b_given_data[event] * data[event][1]])

        # Find New total and heads for coin A
        total_A = 0
        heads_A = 0
        for event in range(len(coinA_heads_tails)):
                total_A += coinA_heads_tails[event][0]+coinA_heads_tails[event][1]
                heads_A += coinA_heads_tails[event][0]

        # Find New total and heads for coin B
        total_B = 0
        heads_B = 0
        for event in range(len(coinB_heads_tails)):
                total_B += (coinB_heads_tails[event][0]+coinB_heads_tails[event][1])
                heads_B += coinB_heads_tails[event][0]

        new_p_a = heads_A / total_A
        new_p_b = heads_B / total_B

        # Run until convergence
        if p_a == new_p_a and p_b == new_p_b:
                return (p_a, p_b)

        else:
                return parameter_estimation(parameters = [new_p_a, new_p_b],
                                            data = data,
                                            prior = prior)



print('initial parameters of coin A and B respectively: ({:},{:})'.format(
        p_heads_a,
        p_heads_b))
print('Final Parameters of coin A and B respectively: {:}'.format(
        parameter_estimation(parameters = [p_heads_a, p_heads_b],
                             data = DATA,
                             prior = p_class)))
