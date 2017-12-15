clc;    % Clear the command window.
close all;  % Close all figures 
clear;  % Erase all existing variables. 

driveTracking = csvread('sampleCrash.csv',2,0);

filename = 'sampleCrash.gif';

x = driveTracking(:,2);
y = driveTracking(:,3);
crash = driveTracking(:,7);

f = figure;
set(f, 'doublebuffer', 'on'); % For smoothness in the animation
 %[make calls to set up plot axes or do first plot]
 axis([-100,100,-50,150])
hold on;
crashYet=0;

for i=1:length(x)
   pause(0.5) % seconds. Adjust to desired speed
   if crash(i)==0
    plot(x(1:i),-y(1:i),'o--g')
   else
    if crashYet == 0
        plot(x(1:i),-y(1:i),'o--g') 
        crashYet=1;
    else
        plot(x(1:i),-y(1:i),'o--r') 
    end
   end
   plot(0,0,'^ k','MarkerSize', 10)
   legend('Pedestrian', 'Car')
   drawnow;
   
   frame = getframe(1);
   im = frame2im(frame);
   [imind,cm] = rgb2ind(im,256);
   if i == 1;
       imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
   else
       imwrite(imind,cm,filename,'gif','WriteMode','append');
   end
   
end

