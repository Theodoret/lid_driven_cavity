clear all;close all;clf;clc;

X   = readmatrix('X.txt');
Y   = readmatrix('Y.txt');
u   = readmatrix('u.txt');
v   = readmatrix('v.txt');
psi = readmatrix('psi.txt');
omega = readmatrix('omega.txt');

if (X(2,1) - X(1,1) == 0 & Y(1,2) - Y(1,1) == 0)
    x = X(1,:); y = Y(:,1);
elseif (X(1,2) - X(1,1) == 0 & Y(2,1) - Y(1,1) == 0)
    x = X(:,1); y = Y(1,:);
    X = X'; Y = Y'; u = u'; v = v'; psi = psi'; omega = omega';
else
    return;
end
Len = sqrt(u.^2+v.^2+1e-10);

figure(1); contourf(x,y,psi,12); axis equal; title('stream function');
figure(2); contourf(x,y,omega,100); axis equal; title('vorticity');
figure(3); quiver(x,y,u./Len,v./Len,1) ;  axis equal; title('veloctiy');