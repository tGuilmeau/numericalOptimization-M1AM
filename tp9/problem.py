import numpy as np



def u(t):
    if t < 0.5:
        res = np.sqrt(t)
    else:
        res = 0.5 + 2*np.abs(t - 0.75)

    return res


N = 500
h = 1.0/N

x_hat = np.array([u(t) + 0.1*np.random.normal() for t in np.linspace(0.0,1.0,N+1)])


diag = (h/2)*np.ones(N+1)
diag[0] = diag[0] / 2
diag[N] = diag[N] / 2
D = np.diag(diag)
invD = np.linalg.inv(D)

C = np.zeros((N,N+1))
for i in range(N):
    C[i,i] = -1
    C[i,i+1] = 1

alpha = 0.001

def f(x):
    return 0.5*np.inner(x - x_hat, D@(x-x_hat)) + alpha*np.linalg.norm(C@x,ord=1)


def g_1(lmbd):
    return 0.5*np.inner(lmbd, (C@invD@C.T)@lmbd) + np.inner(C@x_hat,lmbd)

def grad_g_1(lmbd):
    return (C@invD@C.T)@lmbd + C@x_hat


def from_lambda_to_x(lmbd):
    return x_hat + invD @ C.T @ lmbd


### TO BE COMPLETED


# Case of a beta = 2 regularizer
beta = 2

diag = np.ones(N-1)
diag = np.hstack((np.array([0.5]), diag, np.array([0.5])))
A1 = 0.5*np.diag(diag)
A2 = 2*np.diag(diag)

for i in range(0,N):
    A2[i,i+1] = -1

for i in range(1,N+1):
    A2[i,i-1] = -1


A = 2 * ( h*A1 + (alpha/beta)*(h**(1-beta))*A2 )
b = -2*(h*A1 @ x_hat)
c = np.inner(x_hat, h*A1@x_hat)

def f_beta2(x):
    return 0.5*np.inner(x, A@x) + np.inner(b,x) + c

def grad_f_beta2(x):
    return A@x + b
