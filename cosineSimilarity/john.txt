import random
a=list(map(int,input().split()))
mm=len(a)//2
mins=[]
for i in range(1,mm**mm):
    a1=random.sample(a,mm)

    a2=[]
    for i in a:
        if i not in a1:
            a2.append(i)
    print(a1,a2)
    print("Sum : ",sum(a1),sum(a2))

    z1=sum(a1)
    z2=sum(a2)
    zz=abs(z1-z2)
    print("diff :",zz)
    mins.append([zz,a1,a2])
    print('-'*25)
# print(mins)

print("lest min value from all iterations and its pairs :",min(mins))