clc;
clear;
A=4;
length=700;
length_2=1399;
Data = randi ([0,1], 500, 101); %generation of random data bits in 500 waveforms  (101 bit each realization)
 
%% Definition of each line code
 
Uni_Polar=Data*A; % maping for 0 to be 0, 1 to be A
Polar_NRZ=((2*Data)-1)*A; % maping for 0 to be –A, 1 to be A
Polar_RZ=((2*Data)-1)*A; % maping for 0 to be –A, 1 to be A
 
Uni_Polar=repelem(Uni_Polar, 1, 7); % to repeat each bit 7 times as it is sampled seven times
Polar_NRZ=repelem(Polar_NRZ, 1, 7);
Polar_RZ=repelem(Polar_RZ, 1, 7);
 
for i=1:500 % Polar_RZ to make sample 5,6,7 to be zero for each bit  
    for j=5:7:707
        Polar_RZ(i,j:j+2)= 0;
    end
end
        
%% Randomization of initial time shift
delay = randi([0 6],500,1); % generation of random delay
uni_polar_delayed = zeros(500,700); 
Polar_NRZ_delayed = zeros(500,700);
Polar_RZ_delayed = zeros(500,700);
for i = 1:500   % take 100 bits (700 samples) with random initial start and random bits 
    uni_polar_delayed(i,:) = Uni_Polar(i, 1+delay(i) : 700+delay(i));
    Polar_NRZ_delayed(i,:) = Polar_NRZ(i, 1+delay(i) : 700+delay(i));
    Polar_RZ_delayed(i,:)=Polar_RZ(i, 1+delay(i): 700+delay(i));
end
 
figure('Name','DAC outputs of unipolar linecode (3 random waveform from ensemble)');
subplot(3,1,1);
stairs(uni_polar_delayed(1, 1:length),'r');
title('Uni Polar Dac out');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
subplot(3,1,2);
stairs(uni_polar_delayed(3, 1:length),'r');
title('Uni Polar Dac out');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
subplot(3,1,3);
stairs(uni_polar_delayed(5, 1:length),'r');
title('Uni Polar Dac out');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
figure('Name','DAC outputs of Polar NRZ  linecodes (3 random waveform from  ensemble)'); 
subplot(3,1,1);
stairs(Polar_NRZ_delayed(1, 1:length),'r');
title('Polar NRZ Dac out ');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
subplot(3,1,2);
stairs(Polar_NRZ_delayed(3, 1:length),'r');
title('Polar NRZ Dac out ');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
subplot(3,1,3);
stairs(Polar_NRZ_delayed(5, 1:length),'r');
title('Polar NRZ Dac out ');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
figure('Name','DAC outputs of Polar RZ  linecodes (3 random waveform from  ensemble)'); 
 
subplot(3,1,1);
stairs(Polar_RZ_delayed(1, 1:length),'r');
title('Polar RZ Dac out ');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
subplot(3,1,2);
stairs(Polar_RZ_delayed(3, 1:length),'r');
title('Polar RZ Dac out ');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
subplot(3,1,3);
stairs(Polar_RZ_delayed(5, 1:length),'r');
title('Polar RZ Dac out ');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
figure('Name','DAC outputs of All  linecodes (3 random waveform from  ensemble)'); 
subplot(3,1,1);
stairs(uni_polar_delayed(1, 1:length),'r');
title('Uni Polar Dac out');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
subplot(3,1,2);
stairs(Polar_NRZ_delayed(1, 1:length),'r');
title('Polar NRZ Dac out ');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
subplot(3,1,3);
stairs(Polar_RZ_delayed(1, 1:length),'r');
title('Polar RZ Dac out ');
xlabel('time');
ylabel('Amplitude');
ylim([-5, 5]);
 
%% Calculate statistical mean
Unipolar_NRZ_Mean = zeros(1,700);
Polar_NRZ_Mean = zeros(1,700);
Polar_RZ_Mean = zeros(1,700); 
for i = 1 : 700  % we take each time sample and sum across all realizations and store it in an array
    Unipolar_NRZ_Mean(1,i) = sum(uni_polar_delayed(:,i)); 
    Polar_NRZ_Mean(1,i)    =sum(Polar_NRZ_delayed(:, i));
    Polar_RZ_Mean(1,i)     =sum(Polar_RZ_delayed(:, i));
end
Unipolar_NRZ_Mean = Unipolar_NRZ_Mean/500; % then we take divide across number of realizations (500)
Polar_NRZ_Mean = Polar_NRZ_Mean/500;
Polar_RZ_Mean = Polar_RZ_Mean/500;
figure('Name','Statistical Mean of each line code');
subplot(3,1,1);
stairs(Unipolar_NRZ_Mean(1, 1:length),'b');
title('Mean Uni Polar ');
xlabel('time');
ylabel('Mean');
ylim([-3, 3]);
 
subplot(3,1,2);
stairs(Polar_NRZ_Mean(1, 1:length),'b');
title(' Mean Polar NRZ  ');
xlabel('time');
ylabel('Mean');
ylim([-3, 3]);
 
subplot(3,1,3);
stairs(Polar_RZ_Mean(1, 1:length),'b');
title(' Mean Polar RZ  ');
xlabel('time');
ylabel('Mean');
ylim([-3, 3]);
 
%% statistical autocorrelation
Rx_Unipolar = zeros(3,700);
Rx_Polar_NRZ = zeros(3,700);
Rx_Polar_RZ = zeros(3,700);
 
k=1;
for i= 350:-1:348
for n = -(i-1):350+(350-i)  % tau is used to represent time shift between each sample and the other
for j= 1:500 % at each each tau we calculate the auto correlation by multiplying each value of R.V. at sample to the corresponding value of R.V. at sampled shifted by tau 
% after multiplication,  sum and store in array 
Rx_Unipolar(k,n+i) = Rx_Unipolar(k,n+i)+(uni_polar_delayed(j,i)*uni_polar_delayed(j,n+i));
Rx_Polar_NRZ(k,n+i) =Rx_Polar_NRZ(k,n+i)+(Polar_NRZ_delayed(j,i)*Polar_NRZ_delayed(j,n+i));
Rx_Polar_RZ(k,n+i) = Rx_Polar_RZ(k,n+i)+(Polar_RZ_delayed(j,i)*Polar_RZ_delayed(j,n+i));
end
end
k=k+1;
end
 
Rx_Unipolar= Rx_Unipolar/500; % Taking average bit  divide  
Rx_Polar_NRZ=Rx_Polar_NRZ/500;
Rx_Polar_RZ=Rx_Polar_RZ/500;
 
figure('Name','Statistical Auto correlation of Uni Polar');
for i= 1:3
subplot(3,1,i);
time = -(349-(i-1)):350+i-1;
plot(time,Rx_Unipolar(i,:));
title('Stats Auto correlation Uni Polar ');
xlabel('time');
ylabel('RX');
ylim([3, 9]);
end
figure('Name','Statistical Auto correlation of Polar NRZ');
for i =1:3
subplot(3,1,i);
time = -(349-(i-1)):350+i-1;
plot(time,Rx_Polar_NRZ(i,:));
title(' Stats Auto correlation Polar NRZ  ');
xlabel('time');
ylabel('RX');
ylim([-5,16]);
    
end
figure('Name','Statistical Auto correlation of Polar RZ');
for i =1:3
subplot(3,1,i);
time = -(349-(i-1)):350+i-1;
plot(time,Rx_Polar_RZ(i,:));
title(' Stats Auto correlation Polar RZ  ');
xlabel('time');
ylabel('RX');
ylim([-5,16]);
    
end
 
 
 
 
%% Time mean
    
Unipolar_NRZ_Time_Mean = zeros(1,500);
 Polar_NRZ_Time_Mean= zeros(1,500);
 Polar_RZ_Time_Mean= zeros(1,500);
for i=1:500
Unipolar_NRZ_Time_Mean(1,i) = sum(uni_polar_delayed(i,:)); % we take each realization and sum across all samples (discrete time)  and store it in an array
Polar_NRZ_Time_Mean(1,i) = sum(Polar_NRZ_delayed(i, :));
Polar_RZ_Time_Mean(1,i) = sum(Polar_RZ_delayed(i, :));
end
 
Unipolar_NRZ_Time_Mean = Unipolar_NRZ_Time_Mean/700; % divide on the number of samples (700 samples) 
Polar_NRZ_Time_Mean = Polar_NRZ_Time_Mean/700;
Polar_RZ_Time_Mean = Polar_RZ_Time_Mean/700;
 
figure('Name','Time Mean of each line code');
subplot(3,1,1);
stairs(Unipolar_NRZ_Mean(1, 1:length),'b');
title('Mean Uni Polar ');
xlabel('time');
ylabel('Mean');
ylim([-3, 3]);
 
subplot(3,1,2);
stairs(Polar_NRZ_Mean(1, 1:length),'b');
title(' Mean Polar NRZ  ');
xlabel('time');
ylabel('Mean');
ylim([-3, 3]);
 
subplot(3,1,3);
stairs(Polar_RZ_Mean(1, 1:length),'b');
title(' Mean Polar RZ  ');
xlabel('time');
ylabel('Mean');
ylim([-3, 3]);
 
%% Time Autocorrelation
Unipolar_NRZ_Time_Auto_corr= zeros(1,700);
polar_NRZ_Time_Auto_corr=zeros(1,700);
polar_RZ_Time_Auto_corr=zeros(1,700);
for t=-349:350
    A= circshift(uni_polar_delayed(1,:), t); % 
    B= circshift(Polar_NRZ_delayed(1,:), t);
    C= circshift(Polar_RZ_delayed(1,:), t);
    Unipolar_NRZ_Time_Auto_corr(1,t+350)=sum(uni_polar_delayed(1,:).*A);
    polar_NRZ_Time_Auto_corr(1,t+350)=sum(Polar_NRZ_delayed(1,:).*B);
    polar_RZ_Time_Auto_corr(1,t+350)=sum(Polar_RZ_delayed(1,:).*C);
end
Unipolar_NRZ_Time_Auto_corr=Unipolar_NRZ_Time_Auto_corr/700;
polar_NRZ_Time_Auto_corr = polar_NRZ_Time_Auto_corr/700;
polar_RZ_Time_Auto_corr = polar_RZ_Time_Auto_corr/700;
figure
subplot(3,1,1)
plot(-349:350,Unipolar_NRZ_Time_Auto_corr(1,:));
title(' Time Auto correlation Uni Polar  ');
xlabel('time');
ylabel('RX');
ylim([3, 9]);
subplot(3,1,2)
plot(-349:350,polar_NRZ_Time_Auto_corr(1,:));
title(' Time Auto correlation Polar NRZ  ');
xlabel('time');
ylabel('RX');
ylim([-5, 16]);
subplot(3,1,3)
plot(-349:350,polar_RZ_Time_Auto_corr(1,:));
title(' Time Auto correlation Polar RZ  ');
xlabel('time');
ylabel('RX');
ylim([-5, 10]);
%% PSD
Fs = 100;
Rx_Unipolar_fft = fft(Rx_Unipolar(3,:))/Fs; %to get power spectral density (PSD)
figure
subplot(4,1,1)
f1=(-length/2:length/2-1)*Fs/length; % here to make the message centerd at zero
plot(f1,abs(fftshift(Rx_Unipolar_fft)));
title('Unipolar PSD');
xlabel('frequency HZ');
ylabel('PSD');
 
Rx_Polar_NRZ_fft = fft(Rx_Polar_NRZ(3,:))/Fs;
subplot(4,1,2)
plot(f1,abs(fftshift(Rx_Polar_NRZ_fft)));
title('polar NRZ PSD');
xlabel('frequency HZ');
ylabel('PSD');
 
Rx_Polar_RZ_fft = fft(Rx_Polar_RZ(3,:))/Fs;
subplot(4,1,3)
plot(f1,abs(fftshift(Rx_Polar_RZ_fft)));
title('polar RZ PSD');
xlabel('frequency HZ');
ylabel('PSD');
 
subplot(4,1,4)
plot(f1,abs(fftshift(Rx_Unipolar_fft)));
title('Unipolar PSD (zoom in)');
xlabel('frequency HZ');
ylabel('PSD');
ylim([0,1]);
