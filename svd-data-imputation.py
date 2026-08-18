import numpy as np
import pandas as pd
import os
os.chdir(r"C:\Users\AARINI\Downloads")
data=pd.read_csv(r"C:\Users\AARINI\Downloads\GTPvar.csv",index_col=0)
data['NApresent']=data.isnull().sum(axis=1)
print(data['NApresent'].value_counts())
df=data[data.NApresent==0]#to find relationship btw variables when no nan present
df=df.drop('NApresent',axis=1)
print(df)
#for rank of matrixn so convert to numpy array first
dfmat=df.to_numpy()
print(np.linalg.matrix_rank(dfmat))
v,s,u=np.linalg.svd(dfmat.T)
print("--- Singular Values (s) ---")
print(np.round(s, 4))
print("\n--- Matrix V ---")
print(np.round(v, 4))
print("\n--- Matrix U Shape ---")
print(u.shape)
tol=1e-8
rank=min(dfmat.shape)-np.abs(s)[::-1].searchsorted(tol)#removing columns lesser than tolerance
A=v[:,rank:]
A=A.T
print(A)
len(data)
len(A)
for i in range(0,len(data)):
    if((data.iloc[i,5]==0)|(data.iloc[i,5]>len(A))):
        continue
    else:
        eqnsneeded=data.iloc[i,5]
        aID = np.empty(0, dtype='int64') # Missing column positions
        bID = np.empty(0, dtype='int64') # Known column positions
        for j in range(len(data.columns) - 1):
           if (pd.isnull(data.iloc[i, j])):
              aID = np.append(aID, j)
        else:
             bID = np.append(bID, j)
             a = A[0:eqnsneeded, aID]
             a = np.array(a)
             x1 = ((data.iloc[i, bID].to_numpy()))
             b2 = -A[0:eqnsneeded, bID]
             b = np.dot(b2, x1)
             x = np.linalg.solve(a, b)
             data.iloc[i, aID] = x