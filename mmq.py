import numpy as np
import matplotlib.pyplot as plt


plt.style.use([
  'science',
  'ieee',
  'grid',
])


def least_squares(x, y, degree):
  X = np.ones((len(x), degree + 1))

  for i in range(len(x)):
    pot = degree
    for j in range(degree):
      X[i, j] = x[i] ** pot
      pot -= 1

  TH = np.linalg.inv(X.T @ X) @ X.T @ y

  return TH


T4_X = np.array([
  20992,
  20480,
  19968,
  19456,
  18688,
  18176,
  17664,
  17158,
  16640,
  16256,
  15744,
  15232,
  14720,
  14208,
  13696,
  13312,
  12800,
  12288,
  11776,
  11264,
  10752,
  10240,
  9856
])

T4_Y = np.array([
  63.3,
  61.4,
  59.2,
  56.9,
  54.3,
  52.1,
  50,
  48,
  46,
  44.2,
  42.2,
  40,
  38,
  36,
  33.6,
  32.1,
  30,
  28,
  26,
  24.1,
  22,
  20,
  18.2
])

T3_X = np.array([
  22272,
  21888,
  21376,
  20736,
  20224,
  19712,
  19072,
  18432,
  17920,
  17280,
  16640,
  16256,
  15616,
  14976,
  14336,
  13824,
  13112,
  12800,
  12160,
  11648,
  11136,
  10496,
  9856,
  9344,
  8704
])

T3_Y = np.array([
  58.7,
  57.2,
  55.4,
  53.3,
  51.4,
  49.3,
  47.2,
  45,
  42.9,
  41,
  38.9,
  37.5,
  35,
  32.9,
  30.9,
  29,
  27,
  25,
  23,
  21,
  19.3,
  17,
  15,
  13,
  11
])

data = {
  'T4': {
    'X': T4_X,
    'Y': T4_Y
  },
  'T3': {
    'X': T3_X,
    'Y': T3_Y
  }
}


# find the correlation coefficient
def corr_coefficient(x, y):
  return np.corrcoef(x, y)[0, 1]


print(' ')
print('T3: corr_coefficient: ', corr_coefficient(T3_X, T3_Y))
print('T4: corr_coefficient: ', corr_coefficient(T4_X, T4_Y))
print(' ')


# find the test value
# T = R * sqrt(n - 2 / (1 - R^2))
def test_value(x, y):
  R = corr_coefficient(x, y)
  n = len(x)
  return R * np.sqrt((n - 2) / (1 - R ** 2))


print(' ')
print('T3: test_value: ', test_value(T3_X, T3_Y))
print('T4: test_value: ', test_value(T4_X, T4_Y))
print(' ')


def t_score(x, y, slope, regression):
  x_hat = np.mean(x)
  n = len(x)

  # standard error of the slope = sqrt [ (1 / (n-2)) * (sum((yi - y_hat)**2) / sum((xi - x_hat)**2)) ].

  y_sum = np.sum((y - regression) ** 2)
  x_sum = np.sum((x - x_hat) ** 2)

  standard_error = np.sqrt((1 / (n - 2)) * (y_sum / x_sum))

  return (slope / standard_error), standard_error


TH_T3 = least_squares(T3_X, T3_Y, degree=1)
TH_T4 = least_squares(T4_X, T4_Y, degree=1)

T3_REGRESSION = TH_T3[0] * T3_X + TH_T3[1]
T4_REGRESSION = TH_T4[0] * T4_X + TH_T4[1]

print(' ')
print('T3: t-score | standard error: ', t_score(T3_X, T3_Y, TH_T3[0], T3_REGRESSION))
print('T4: t-score | standard error: ', t_score(T4_X, T4_Y, TH_T4[0], T4_REGRESSION))
print(' ')

print(' ')
print('T3: a={}, b={}'.format(TH_T3[0], TH_T3[1]))
print('T4: a={}, b={}'.format(TH_T4[0], TH_T4[1]))
print(' ')

plt.figure(figsize=(10, 5))

plt.plot(T3_X, T3_Y, 'o', label='T3')
plt.plot(T4_X, T4_Y, 'o', label='T4')

plt.plot(T3_X, T3_REGRESSION, '--', label='T3 approximation')
plt.plot(T4_X, T4_REGRESSION, '--', label='T4 approximation')
plt.legend(['T3', 'T4', 'T3 approximation', 'T4 approximation'])

plt.xlabel('Sensor reading')
plt.ylabel('Position (Cm)')

plt.savefig('mmq.png', dpi=300)
plt.show()
