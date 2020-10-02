import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


er=[]
m1=[]
c1=[]
te=[]

def compute_error(c,m,x,y):
    totalError=0
    x=x
    y=y
    
    m1.append(m)
    
    c1.append(c)
    
    for i in range(0,len(x)):
        
        totalError=totalError+(((m*x[i]+c)-y[i])**2)
        
    er.append(float(totalError/len(x)))
    


def step_gradient(c_current,m_current,x,y,learningRate):
    c_gradent=0.0
    m_gradent=0.0
    N=float(len(x))
    x=x
    y=y
    cg=[]
    mg=[]
    for i in range(0,len(x)):
        cg.append(((m_current*x[i])+c_current)-y[i])
        mg.append(x[i]*(((m_current*x)+c_current)-y[i]))
        
    c_gradent=(np.array(cg).sum())/N
    m_gradent=(np.array(mg).sum())/N
    
    new_c=c_current-(learningRate*c_gradent)
    new_m=m_current-(learningRate*m_gradent)
    
    compute_error(new_c,new_m,x,y)
    
    return[new_c,new_m]


def gradient_descent_runner(x,y,starting_c,starting_m,learning_rate,num_iterations):
    c=starting_c
    m=starting_m
    for i in range(num_iterations):
        c,m=step_gradient(c,m,x,y,learning_rate)
        
        


def run():
    sn=pd.read_excel(r'C:\Users\shiva sk\Downloads\salary\Salary1_Data.xlsx')
    x1=sn.iloc[:,0:1]
    y1=sn.iloc[:,1:2]
    
    x=np.array(x1)
    y=np.array(y1)
    
    learning_rate=0.0003
    initial_c=1#initial x_intercept guess
    initial_m=1#initial y_slope guess
    num_iteration=10
    compute_error(initial_c,initial_m,x,y)
    gradient_descent_runner(x,y,initial_c,initial_m,learning_rate,num_iteration)
    return x1,y1



u,v=run()
it=np.arange(0,11)

z=er[0]
for i in er:
    if z>i:
        z=i
#print('minimum  er =',z)
q=er.index(z)
print('iteration =',q)
cc=c1[q]
mm=m1[q]

plt.plot(u,v,'ro',u,u*mm+cc)
plt.show()

print('approximate  c =',c1[q])
print('approximate  m =',m1[q])
print('\n\n')
    
plt.plot(it,er)
plt.show()    

print(pd.DataFrame({'Years Experience':[4.2,5.2],'Salary':[int(4.2*mm+cc),int(5.2*mm+cc)]}))
