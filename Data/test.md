# 旋转矩阵，sin的负号在上还是下

谈一谈自己对旋转矩阵的理解。主要就是关系到sin的负号在上还是在下。

首先，逆时针都为正（右手定则的方向），顺时针为负。

那么，其实sin负号在上还是在下对应着两种理解旋转矩阵的方式，分别表示视角（坐标系）的变换和物体的变换。注意，另一种说法是顺时针逆时针差了个负号，这个很好理解，在此不赘述。

因为我的习惯是sin负号在上比较顺延，所以首先定义

$$
R=\begin{bmatrix}
\cos(\theta)&-\sin(\theta)\\
\sin(\theta)&\cos(\theta)
\end{bmatrix}
$$

# 视角变换

**第一种旋转矩阵是表示视角，即坐标系的变换，记作$R_{coord}$。**

就是说空间有个点P，它是固定不变的。你建立了一个坐标系C1，现在把坐标系转了$\theta$到C2，那么P在C1下的坐标和P在C2下的坐标有如下关系：

$$
P_{C1}=
\begin{bmatrix}
\cos(\theta)&-\sin(\theta)\\
\sin(\theta)&\cos(\theta)
\end{bmatrix}
P_{C2}=RP_{C2}，即\\
P_{C2}=R^{-1}P_{C1}=\begin{bmatrix}
\cos(\theta)&\sin(\theta)\\
-\sin(\theta)&\cos(\theta)
\end{bmatrix}P_{C1}=R_{coord}P_{C1}
$$

这个可以通过矩阵乘法和基向量的概念来推导：

首先要明确，点和向量本身并无坐标，我们要选坐标系对它们进行描述。P在坐标系C下的坐标为$(x,y)$的含义是，构成C的基向量的一个线性组合，即$x\vec{i}+y\vec{j}$描述了P的空间位置。

为了方便记，我们将当前坐标系的基向量坐标写成矩阵的列，作为矩阵$A$，将坐标写成列向量$x=[x,y]^T$，则因为矩阵乘法$Ax$表示的是用x的行对A的列进行线性组合，所以有

$$
P_{C}=
\begin{bmatrix}
\space \vec{i}&\vec{j} \space 
\end{bmatrix}
\begin{bmatrix}
x\\y
\end{bmatrix}=x\vec{i}+y\vec{j}
$$

也就是说，$P_C$表示的就是坐标为$(x,y)$的向量（点）在C坐标系下的位置。

此外，因为我们写矩阵的过程中要用到基向量的坐标，而凡是坐标，都要有坐标系，于是可以认为存在一个最基准的坐标系C，以后所有基向量的坐标就是在这个系里描述的。

这样的话，假设我们有另一组基向量构成了坐标系$C'$。设有一个点$Q=(x',y')$，那么$Q$在$C'$中的坐标是$[x,y]^T$，而$Q$在**基准坐标系C中**的描述就是在基准坐标系C中的两个构成$C'$的基向量的线性组合。根据上面的定义，是

$$
Q_{C}=
\begin{bmatrix}
\space \vec{i'}&\vec{j'} \space 
\end{bmatrix}
\begin{bmatrix}
x'\\y'
\end{bmatrix}=x'\vec{i'}+y'\vec{j'}
$$

知道这些就可以来推导旋转矩阵了。

对于变换后的坐标系C2，因为点的空间位置不变，也就是说$P_{C1}$不变。设$\vec{i_{C2}}，\vec{j_{C2}}$ 表示C2的基向量在基准坐标系中的坐标，那么有

$$
\begin{bmatrix}
\space \vec{i_{C2}}&\vec{j_{C2}} \space 
\end{bmatrix}
\begin{bmatrix}
x_{C2}\\y_{C2}
\end{bmatrix}=P_{C1}，即\\
R*P_{C2}=P_{C1}
$$

那么R的求法正如上文所说：即旋转后C2的基向量在C1中的坐标，画一个简单的逆时针旋转的图即可知道，

$$
i_{C2}=[\cos(\theta), \sin(\theta)]^T，j_{C2}=[-\sin(\theta),\cos(\theta)]^T
$$

写成矩阵的列的形式就是

$$
R=\begin{bmatrix}
\cos(\theta)&-\sin(\theta)\\
\sin(\theta)&\cos(\theta)
\end{bmatrix}
$$

而表示这个坐标变换的矩阵就是它的逆，

$$
R_{coord}=R^{-1}=\begin{bmatrix}
\cos(\theta)&\sin(\theta)\\
-\sin(\theta)&\cos(\theta)
\end{bmatrix}
$$

这就是所谓的第一种形式，表示坐标系的转换，可以理解为我们的视角的变换——点不变，我们从另一个视角观察它得到的坐标。

# 刚体变换

**第二种是表示物体的变换。**

$$
R=\begin{bmatrix}
\cos(\theta)&-\sin(\theta)\\
\sin(\theta)&\cos(\theta)
\end{bmatrix}
$$

意思是，当旋转坐标系C的时候，把点跟着转。可以想象为P在一个初始与C1重合的刚体上，现在我把刚体转了$\theta$，P也会跟着转。那么在刚体的新位置处建立一个新的坐标系C2，求P在原先坐标系C1下的坐标，我们先给出形式，就是

$$
P_{C1}=
\begin{bmatrix}
\cos(\theta)&-\sin(\theta)\\
\sin(\theta)&\cos(\theta)
\end{bmatrix}
P_{C2}
$$

可以发现$R_{coord}=R^T$，又因为旋转矩阵是正交阵，即$R^T=R^{-1}$，所以这种和第一种是恰好差了一个逆，$R_{coord}=R^{-1}$。

那么这种是怎么推导的呢？其实可以转换为第一种，即：可以认为是点没动，坐标系C1向反方向转了$\theta$的结果。

![Untitled](%E6%97%8B%E8%BD%AC%E7%9F%A9%E9%98%B5%EF%BC%8Csin%E7%9A%84%E8%B4%9F%E5%8F%B7%E5%9C%A8%E4%B8%8A%E8%BF%98%E6%98%AF%E4%B8%8B%2054e48c8b94f64b4c96801a9b0f6f40f1/Untitled.png)

那么现在的问题转化为：空间有个点P，它是固定不变的。你建立了一个坐标系C1，现在把坐标系转了$-\theta$转到C2，求在C2下的坐标？由第一种矩阵的含义，有

$$
P_{C1}=
\begin{bmatrix}
\cos(-\theta)&-\sin(-\theta)\\
\sin(-\theta)&\cos(-\theta)
\end{bmatrix}
P_{C2}=
\begin{bmatrix}
\cos(\theta)&\sin(\theta)\\
-\sin(\theta)&\cos(\theta)
\end{bmatrix}
P_{C2}=R^{-1}P_{C2}
$$

即：$P_{C2}=RP_{C1}$。也就是说在原坐标系C下，对$P$进行了旋转$\theta$的操作后，坐标是$RP$。即表示刚体变换，而不是坐标系变换。

总结一下，如果表示点不动，坐标系变了的话，就用负号在下面的，

$$
R_{coord}=\begin{bmatrix}
\cos(\theta)&\sin(\theta)\\
-\sin(\theta)&\cos(\theta)
\end{bmatrix}
$$

如果表示坐标系不变，对物体进行旋转就用：

$$
R=\begin{bmatrix}
\cos(\theta)&-\sin(\theta)\\
\sin(\theta)&\cos(\theta)
\end{bmatrix}
$$

# 应用

## 1.SVD中对变换分解的理解

协方差矩阵的SVD中，有（以二维举例）

$$
C=R\Lambda R^{-1}=R\begin{bmatrix}
\sigma_1 & 0\\
0 & \sigma_2
\end{bmatrix} R^{-1}
$$

R是旋转矩阵。

实际上意思是，如果数据分布是一个椭圆形状，那么我们可以找到椭圆的长轴和短轴方向的伸缩，分别是$\sigma_1，\sigma_2$。

意思是，对于一个单位圆，我们先把坐标轴转R，再在新的坐标轴x，y方向分别伸缩$\sigma_1，\sigma_2$，然后再把坐标轴转回去，这样就得到了上述椭圆。

比如下面这个椭圆，若已知长轴的方向和y轴夹角是30度，且长短轴之比是2：1，如何由一个圆得到它呢？

![Untitled](%E6%97%8B%E8%BD%AC%E7%9F%A9%E9%98%B5%EF%BC%8Csin%E7%9A%84%E8%B4%9F%E5%8F%B7%E5%9C%A8%E4%B8%8A%E8%BF%98%E6%98%AF%E4%B8%8B%2054e48c8b94f64b4c96801a9b0f6f40f1/Untitled%201.png)

**因为转的是坐标轴**，所以答案是

$$
C=R_{\pi/6}^{-1}\begin{bmatrix}
1 & 0\\
0 & 2\\
\end{bmatrix}R_{\pi/6}=\begin{bmatrix}
\cos(\frac{\pi}{6}) & -\sin(\frac{\pi}{6})\\
\sin(\frac{\pi}{6}) & \cos(\frac{\pi}{6})\\
\end{bmatrix}\begin{bmatrix}
1 & 0\\
0 & 2\\
\end{bmatrix}\begin{bmatrix}
\cos(\frac{\pi}{6}) & \sin(\frac{\pi}{6})\\
-\sin(\frac{\pi}{6}) & \cos(\frac{\pi}{6})\\
\end{bmatrix}
$$

即先把坐标轴转30度，然后在x方向拉伸1，y方向拉伸2，再转回去。

算出来此矩阵的结果即为此椭圆分布的数据的协方差矩阵：

$$
C=\begin{bmatrix}
1.25  & -0.43\\
-0.43 & 1.75\\
\end{bmatrix}
$$

从直观上来看，它表示数据在x方向散开得更小（$\sigma_x^2=1.25$），在y方向散开得更大（$\sigma_y^2=1.75$），且x和y大致呈现负相关。这和椭圆形状是相符的。

需要注意的是，这个矩阵并不是椭圆的平面方程$x^TMx=d$中的$M$矩阵，实际上，这里用到了旋转矩阵的第二种含义来推导。

单位圆的方程是$x^Tx=d$，那么现在对每个点进行变换（根据上文推导，对坐标轴施以变换$C$等价于对点施以变换$C^{-1}$，因此$x'=Cx$）

那么新的点要变回原先的点就要$x=C^{-1}x'$，带入圆的方程有$x'^T(C^{-T}C^{-1})x'=d$，这就是变换后的点满足的方程，即椭圆方程。带入得

$$
M=C^{-T}C^{-1}≈\begin{bmatrix}
0.8125 & 0.3247\\
0.3247 & 0.4375
\end{bmatrix}
$$

将$x=[x,y]^T$代入，展开得到$0.81x^2+0.43y^2+0.65xy=1$。作图验证如下（尺度不对是因为原图并不是1：2的放缩比）

![Untitled](%E6%97%8B%E8%BD%AC%E7%9F%A9%E9%98%B5%EF%BC%8Csin%E7%9A%84%E8%B4%9F%E5%8F%B7%E5%9C%A8%E4%B8%8A%E8%BF%98%E6%98%AF%E4%B8%8B%2054e48c8b94f64b4c96801a9b0f6f40f1/Untitled%202.png)

附Python代码

```python
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt

theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))
V = np.array([[1, 0],
			  [0, 2]])
print('R=', R)
C = R @ V @ lin.inv(R)
print(C) # 协方差矩阵
M = lin.inv(C).T @ lin.inv(C)
print(M) # 椭圆矩阵

x = y = np.arange(-10, 10, 0.1)
x, y = np.meshgrid(x, y)
plt.contour(x, y,
	M[0,0] * x ** 2 + M[1,1] * y ** 2 + (M[0,1]+M[1,0]) * x * y, [1]) 
  #等高线图，最后一个参数是意思是画出前面的式子等于几的那一条轮廓，如果省略就会画出很多条，这里为1意思是常数项是1（单位圆）
	# Determines the number and positions of the contour lines / regions.
	# If an int n, use MaxNLocator, which tries to automatically choose no more than n+1 "nice" contour levels between minimum and maximum numeric values of Z.
	# If array-like, draw contour lines at the specified levels. The values must be in increasing order.

plt.axis('scaled')
plt.show()
```

## 2. 不同学科中的变换矩阵

机器人学中经常用到的是刚体变换，就是第二种，$R$；而惯导原理中经常要对坐标系进行变换，于是是第一种，$R^{-1}$，并不冲突。

比如，[惯导中](https://blog.csdn.net/Mua111/article/details/125433510#1ie_38)有大地坐标系e到东北天坐标系g的变换

![Untitled](%E6%97%8B%E8%BD%AC%E7%9F%A9%E9%98%B5%EF%BC%8Csin%E7%9A%84%E8%B4%9F%E5%8F%B7%E5%9C%A8%E4%B8%8A%E8%BF%98%E6%98%AF%E4%B8%8B%2054e48c8b94f64b4c96801a9b0f6f40f1/Untitled%203.png)

![Untitled](%E6%97%8B%E8%BD%AC%E7%9F%A9%E9%98%B5%EF%BC%8Csin%E7%9A%84%E8%B4%9F%E5%8F%B7%E5%9C%A8%E4%B8%8A%E8%BF%98%E6%98%AF%E4%B8%8B%2054e48c8b94f64b4c96801a9b0f6f40f1/Untitled%204.png)

g系先绕OZ轴转动-pi/2，接着绕OY轴转动-(pi/2-B)，其中B为纬度（弧度），再绕OZ轴转动-L，其中L为经度（弧度）。可以看到，用的是第一种形式，sin负号在下。

而机器人学中，比如在3D空间中对一辆车进行绕固定系z轴旋转（绕固定系是左乘）$\theta$，就是sin在上的表示方式：

$$
R=\begin{bmatrix}
\cos(\theta)&-\sin(\theta)&0\\
\sin(\theta)&\cos(\theta)&0\\
0 & 0 & 1
\end{bmatrix}
$$