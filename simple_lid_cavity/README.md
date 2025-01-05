A cpp program for solving a simple lid driven cavity numerically, using finite difference and explicit Euler's method.\
The vorticity function is
```math
\frac{\partial \Omega}{\partial t} + u \frac{\partial \Omega}{\partial x} + v \frac{\partial \Omega}{\partial y} = \frac{1}{Re} (\frac{\partial^2 \Omega}{\partial x^2} + \frac{\partial^2 \Omega}{\partial y^2})
```

, the steam function is
```math
\frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial y^2} = \Omega
```

, and the velocity is
```math
\displaylines{
u = \frac{\partial \psi}{\partial y}
\\
v = -\frac{\partial \psi}{\partial x}.
}
```

Meanwhile, the boundary is\
<p align="center">
<img src="https://www.cfd-online.com/W/images/a/a3/Ldc_geom.png" width="" height="" border=""/>
</p>
