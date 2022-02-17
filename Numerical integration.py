# %%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

T = 0.25
K = 12
sigma = 0.2
S = 10
r = 0.01
d_1 = (1 / (sigma * np.sqrt(T))) * (np.log(S/K) + T * (r + sigma ** 2 / 2))
d_2 = (1 / (sigma * np.sqrt(T))) * (np.log(S/K) + T * (r - sigma ** 2 / 2))

def black_scholes():
    return S * norm.cdf(d_1) - K * np.exp(-r*T) * norm.cdf(d_2)
    
def LeftRiemannrule(n):# integral from -inf to d1, -inf to d2
    a = -10
    pdf_d1 = pdf_d2 = 0
    nodes1 = [a + ((d_1 - a) / n) * (i - 1) for i in range(1, n)]
    nodes2 = [a + ((d_2 - a) / n) * (i - 1) for i in range(1, n)]
    for n1 in nodes1:
        pdf_d1 += norm.pdf(n1) * ((d_1 - a) / n)
    for n2 in nodes2:
        pdf_d2 += norm.pdf(n2) * ((d_2 - a) / n)
    return S * pdf_d1 - K * np.exp(-r*T) * pdf_d2
    
def mid_point_rule(n):
    a = -10
    pdf_d1 = pdf_d2 = 0
    nodes1 = [a + ((d_1 - a) / n) * (i - 0.5) for i in range(1, n)]
    nodes2 = [a + ((d_2 - a) / n) * (i - 0.5) for i in range(1, n)]
    for n1 in nodes1:
        pdf_d1 += norm.pdf(n1) * ((d_1 - a) / n)
    for n2 in nodes2:
        pdf_d2 += norm.pdf(n2) * ((d_2 - a) / n)
    return S * pdf_d1 - K * np.exp(-r*T) * pdf_d2

def gaussian(n):
    a = -1
    b = 1
    pdf_d1 = pdf_d2 = 0
    node, weight = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * node * (b - a) - (- b - a) / 2

    slope_d1 = (d_1 + 10) / 2
    intercept_d1 = - (10 - d_1) / 2
    for j in range(len(nodes)):
        pdf_d1 += weight[j] * (1 / np.sqrt(2 * np.pi)) * np.exp(- (slope_d1 * nodes[j] + intercept_d1) ** 2 / 2) * slope_d1
    
    slope_d2 = (d_2 + 10) / 2
    intercept_d2 = - (10 - d_2) / 2
    for j in range(len(nodes)):
        pdf_d2 += weight[j] * (1 / np.sqrt(2 * np.pi)) * np.exp(- (slope_d2 * nodes[j] + intercept_d2) ** 2 / 2) * slope_d2

    return S * pdf_d1 - K * np.exp(-r*T) * pdf_d2

# %%
n = range(4, 100)
left = [LeftRiemannrule(i) for i in n]   
mid  = [mid_point_rule(i) for i in n]  
gauss= [gaussian(i) for i in n]
N_3  = [i ** (-3) for i in n]
N_2  = [i ** (-2) for i in n]
real = black_scholes() 
err_left  = abs(real - left)
err_mid   = abs(real - mid)
err_gauss = abs(real - gauss)
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.plot(n, err_mid, 'r-', lw=3, alpha=0.6, label='Mid point rule error')
ax.plot(n, err_left, 'y-', lw=3, alpha=0.6, label='Left Riemann rule error')
ax.plot(n, err_gauss, 'k-', lw=3, alpha=0.6, label='Gaussian rule error')
ax.plot(n, N_3, 'g-', lw=3, alpha=0.6, label='N^-3')

ax.set_xlabel('Nodes')
ax.set_ylabel('Error')
ax.legend()


# %%
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

sigma1 = 20
sigma2 = 15
k1 = 380
k2 = 375
r = 0
N = 1000
rho = 0.95

def mid_nodes(N, a, b):
    return [a + ((b - a) / N) * (i + 1 / 2) for i in range(N)]

# (a)# 7.978878853509584
weight = ((k1 + 10 * sigma1 - k1) / 10000)
nodes = mid_nodes(10000, k1, k1 + 10 * sigma1)
vanilla = 0
for n in nodes:
    vanilla += (n - k1) * norm.pdf(n, k1, sigma1) * weight

# (b)
def joint_pdf(x, y, sigma1, sigma2, mu1, mu2, rho):
    first = 1 / (2 * np.pi * sigma1 * sigma2 * np.sqrt(1 - rho ** 2))
    later = -0.5 * (1 / (1 - rho ** 2)) * (((x - mu1) / sigma1) ** 2 - (2 * rho * (x - mu1) * (y - mu2)) / (sigma1 * sigma2) + ((y - mu2) / sigma2) ** 2)
    return first * np.exp(later)
def contingent():
    nodea = mid_nodes(N, k1, k1 + 5 * sigma1)
    nodeb = mid_nodes(N, k2 - 5 * sigma2, k2)
    ans = 0
    for i in range(len(nodea)):
        for j in range(len(nodeb)):
            ans += (nodea[i] - k1) * joint_pdf(nodea[i], nodeb[j], sigma1, sigma2, k1, k1, rho) * (((5 * sigma1 * 5 * sigma2) / (N**2)))
    return ans

contingent()



# %% [markdown]
# ## Problem set # 1
# #### 1. What areas of finance 
#    I enjoy doing data analysis for financial time series, trading and research are the best. Before that I studied Physics and learned quite a lot how to do labs and analyze experiment data. Now after a semester study, I found these data analysis pretty systematic and not hard to learn, so I believe I can use these skills to futher extract information from these data. Hopefully, I will try to apply machine learning and everything else that I might be able to use in my work and do more useful things in financial fields.
# #### 2. What is the primary reason for your interest in this course? 
#    I studied computational Physics before, I think I still need to study more about how to do computations in finance. This course so far is great for me becuase I do have learned a lot in this homework. 
# #### 3. List all programming languages you have used
# Python(Most familiar), R code(beginner), C++(A bit better than R), C(Beginner), MATLAB(Intermediate), Julia(Beginner)
# #### 4. Option traders often say that when buying options we get gamma at the expense of theta 
# **Gamma**: $\Gamma = \frac{\partial^{2} f}{\partial S^{2}}$,
# **Theta**: $\Theta = \frac{\partial f}{\partial t}$,
# #### 5. Evaluation of a known integral using various quadratures: 
# (a) Use Black-Scholes formula to compute the price of the call analytically.<p>
#     0.015673906956796213 <p>
# (b) Calculate the price of the call numerically using the following 3 quadrature methods <p>
#     Left Riemann Rule:<p> n = 5: 2.458585063294997e-06, Error: 0.01567144837173292 <p>
#                        n = 10: 0.0010960810989985217, Error: 0.014577825857797691 <p>
#                        n = 50: 0.012126869736813567, Error: 0.003547037219982646 <p>
#     Mid Point Rule:<p>    n = 5: 0.00010769551089158836, Error: 0.015566211445904624 <p>
#                        n = 10: 0.003547508357272487, Error: 0.012126398599523726 <p>
#                        n = 50: 0.01346747123900116, Error: 0.002206435717795052 <p>
#     Gaussian Nodes legendre method: Because it's integrating over the interval [−1, 1] without considering $\infty$. Also, it has package that I can use directly to get the results, so I chose that one. <p>
#                        n = 5: 2.458585063294997e-06,  Error: -0.01567144837173292 <p>
#                        n = 10: 0.0010960810989985217,  Error: -0.014577825857797691 <p>
#                        n = 50: 0.012126869736813567,  Error: -0.003547037219982646 <p>  
# (c) Estimate the experimental rate of convergence (i.e., as a function of N) of each method and compare it with the known theoretical estimate. <p>
#     I plotted a $\frac{1}{N^{-3}}$ curve to make the comparison clearer in the graph. It shows us that the convergence of Gaussian nodes method is faster than other two methods, it's converging similarly in $\frac{1}{N^{-3}}$ rate. 
# (d) I prefer mid point rule because it's much easier to apply and to understand. Though the precision of Gaussian nodes method is better, I think mid point rule is an easy way to use usually without considering too much about the precision and how fast it concerges. 
# 
# 
# 
# 

# %% [markdown]
# ## 6. Calculation of Contingent Options:
# #### (a) Evaluate the price of the one year call on ABC with the strike K1 = 380. This is an example of a vanilla option.
# The price of this vanilla option is 7.978878853509584
# #### (b) Evaluate the price of the one year call on ABC with the strike K1 = 380, contingent on ABC at 6 months being below 375. This is a contingent option.
# The price of this contingent option is 0.026689302978177218
# #### (c) Calculate the contingent option again, but with ρ = 0.8, ρ = 0.5, and ρ = 0.2.
# $\rho=0.8$ : $price = 0.3252550318608471$
# $\rho=0.5$ : $price = 1.1948854660980905$
# $\rho=0.2$ : $price = 2.2132188586278425$
# #### (d) Does dependence on ρ make sense?
# It makes sense, as $\rho$ decreases, the price of this contingent option is increased. As we know, if the correlation is high, both two variables will move the same way, so if we get a lower value in 6 months, we also get a lower value in one year. Thus, if they have high correlation, we cannot expect a high value of the underlying index in one year if it's lower than 375 in 6 months.
# #### (e) Calculate the contingent option again, but with ABC at 6 months below 370 and 360.
# $\rho=0.95, k2 = 370$ : $price = 0.0016136183860122446$
# $\rho=0.95, k2 = 360$ : $price = 3.652588494899704e-07$
# #### (f) Does the dependence on the 6 month value make sense?
# It makes sense, from (e), as k2 decreased, the price of this contingent option also decreased. We can explain that because the lower it reaches in 6 months, the lower probability that it can come back in a high value in one year.
# #### (g) Under what conditions do you think the price of the contingent option will equal the price of the vanilla one?
# I think they are equal when the required value of 6 months condition is tending to the infinity. Since we are restricted by the condition as long as it exists, so we remove it by making it big enough that it doesn't matter for our options. Thus, when it's high enough, we don't need to care about this condition
# 


