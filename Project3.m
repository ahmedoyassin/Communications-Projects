close all
clc;
clear;
%% Data Control
N_bits=24000;
data_bits = randi ([0,1], 1, N_bits); % generate random data
SNR_dB=-4:1:14;
SNR_linear=zeros(1,size(SNR_dB,2)); % Eb/No
No=zeros(1,size(SNR_dB,2)); % Eb/No
for i=1:size(SNR_dB,2)
SNR_linear(i)=10^(SNR_dB(i)/10);
No(i)=1/SNR_linear(i); % Calculate normalized No.
end
counter =length(No); % used to loop on values of SNR
%% BPSK
BPSK=((2*data_bits)-1); % mapping to 1 , -1
E=BPSK.^2;
E_avg_BPSK=sum(E)/N_bits;
%channel effect
BPSK_noise=zeros(counter,N_bits);
Demapped_BPSK=zeros(counter,N_bits);
BER_BPSK_theoritical=zeros(1,counter);
BER_BPSK=zeros(1,counter);
for i= 1:counter
random_noise=randn(1 , N_bits);
Noise_BPSK=sqrt(No(i)/2*E_avg_BPSK)*random_noise;
BPSK_noise(i,:)=BPSK+Noise_BPSK;
for j=1:N_bits %Demapper
if(BPSK_noise(i,j)>=0)
Demapped_BPSK(i,j)=1;
else
Demapped_BPSK(i,j)=0;
end
end
for k=1:N_bits
if data_bits(k) ==1 && Demapped_BPSK(i,k)==0 || data_bits(k) ==0 && Demapped_BPSK(i,k)==1
BER_BPSK(1,i)=BER_BPSK(1,i)+1;
end
end
BER_BPSK_theoritical(i)=0.5*erfc(sqrt(SNR_linear(i)));
end
BER_BPSK=BER_BPSK/N_bits;
figure('Name','BER')
semilogy(SNR_dB,BER_BPSK,'g','Linewidth',3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
hold on
semilogy(SNR_dB,BER_BPSK_theoritical,'b--','Linewidth',3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
legend( 'BER BPSK','BER BPSK theoritical','Location', 'best');
%% Gray_QPSK
Gray_QPSK_constellation=[-1-1i -1+1i 1-1i 1+1i];
reshaped_data = reshape(data_bits, 2, []);
reshaped_data = reshaped_data';
decimal_number = bin2dec(num2str(reshaped_data));
decimal_number = decimal_number';
%mapper
QPSK_gray=zeros(1,N_bits/2);
for i=1:N_bits/2
QPSK_gray(i)=Gray_QPSK_constellation(decimal_number(i)+1);
end
E=abs(QPSK_gray).^2;
E_avg_QPSK_gray=sum(E)/(N_bits*0.5);
%channel effect
QPSK_noise=zeros(counter,N_bits/2);
Demapped_QPSK=zeros(counter,N_bits/2);
Binary_Demapped_QPSK_gray=zeros(counter,N_bits);
BER_QPSK_gray_theoritical=zeros(1,counter);
BER_QPSK_gray=zeros(1,counter);
for i= 1:counter
real_noise=randn(1 , N_bits/2);
imag_noise=randn(1 , N_bits/2);
Noise_QPSK=sqrt(No(i)/2*(E_avg_QPSK_gray/2))*real_noise+1i*sqrt(No(i)/2*(E_avg_QPSK_gray/2))*imag_noise;
QPSK_noise(i,:)=QPSK_gray+Noise_QPSK;
for j=1:N_bits/2
[value,index]=min(abs(QPSK_noise(i,j)-Gray_QPSK_constellation));
Demapped_QPSK(i,j)=index-1;
Binary_value=dec2bin(Demapped_QPSK(i,j),2);
Binary_Demapped_QPSK_gray(i,(2*j-1):(2*j))=str2num(Binary_value(:));
end
for k=1:N_bits
if data_bits(k) ==1 && Binary_Demapped_QPSK_gray(i,k)==0 || data_bits(k) ==0 && Binary_Demapped_QPSK_gray(i,k)==1
BER_QPSK_gray(1,i)=BER_QPSK_gray(1,i)+1;
end
end
BER_QPSK_gray_theoritical(i)=0.5*erfc(sqrt(SNR_linear(i)));
end
BER_QPSK_gray= BER_QPSK_gray/N_bits;
figure('Name','BER')
semilogy(SNR_dB,BER_QPSK_gray,'g','Linewidth',3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
hold on
semilogy(SNR_dB,BER_QPSK_gray_theoritical,'b--','Linewidth',3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
legend( 'BER QPSK gray','BER QPSK gray theoritical','Location', 'best');
%% nogray_QPSK
no_gray_QPSK_constellation=[-1-1i -1+1i 1+1i 1-1i];
reshaped_data2 = reshape(data_bits, 2, []);
reshaped_data2 = reshaped_data2';
decimal_number2 = bin2dec(num2str(reshaped_data2));
decimal_number2 = decimal_number2';
%mapper
QPSK_nogray=zeros(1,N_bits/2);
for i=1:N_bits/2
QPSK_nogray(1,i)=no_gray_QPSK_constellation(decimal_number2(i)+1);
end
E=abs(QPSK_nogray).^2;
E_avg_QPSK_nogray=sum(E)/(N_bits*0.5);
%channel effect
QPSK_noise=zeros(counter,N_bits/2);
Demapped_QPSK_nogray=zeros(counter,N_bits/2);
Binary_Demapped_QPSK_nogray=zeros(counter,N_bits);
BER_QPSK_nogray_theoritical=zeros(1,counter);
BER_QPSK_nogray=zeros(1,counter);
for i= 1:counter
real_noise=randn(1 , N_bits/2);
imag_noise=randn(1 , N_bits/2);
Noise_QPSK=sqrt(No(i)/2*(E_avg_QPSK_nogray/2))*real_noise+1i*sqrt(No(i)/2*(E_avg_QPSK_nogray/2))*imag_noise;
QPSK_noise(i,:)=QPSK_nogray+Noise_QPSK;
for j=1:N_bits/2
[value,index]=min(abs(QPSK_noise(i,j)-no_gray_QPSK_constellation));
Demapped_QPSK_nogray(i,j)=index-1;
Binary_value=dec2bin(Demapped_QPSK_nogray(i,j),2);
Binary_Demapped_QPSK_nogray(i,(2*j-1):(2*j))=str2num(Binary_value(:));
end
for k=1:N_bits
if data_bits(k) ==1 && Binary_Demapped_QPSK_nogray(i,k)==0 || data_bits(k) ==0 && Binary_Demapped_QPSK_nogray(i,k)==1
BER_QPSK_nogray(1,i)=BER_QPSK_nogray(1,i)+1;
end
end
BER_QPSK_nogray_theoritical(i)=0.5*erfc(sqrt(SNR_linear(i)));
end
BER_QPSK_nogray= BER_QPSK_nogray/N_bits;
figure('Name','BER')
semilogy(SNR_dB,BER_QPSK_nogray,'g','Linewidth',3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
hold on
semilogy(SNR_dB,BER_QPSK_nogray_theoritical,'b--','Linewidth',3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
% grid on;
legend( 'BER QPSK nogray','BER QPSK nogray theoritical','Location', 'best');
%% QAM
QAM_constellation=[-3-3i -3-1i -3+3i -3+1i -1-3i -1-1i -1+3i -1+1i 3-3i 3-1i 3+3i 3+1i 1-3i 1-1i 1+3i 1+1i];
reshaped_data3 = reshape(data_bits, 4, []);
reshaped_data3 = reshaped_data3';
decimal_number3 = bin2dec(num2str(reshaped_data3));
decimal_number3 = decimal_number3';
%mapper
QAM=zeros(1,N_bits/4);
for i=1:N_bits/4
QAM(i)=QAM_constellation(decimal_number3(i)+1);
end
E=abs(QAM).^2;
E_avg_QAM=sum(E)/(N_bits*0.25);
%channel effect
QAM_noise=zeros(counter,N_bits/4);
Demapped_QAM=zeros(counter,N_bits/4);
Binary_Demapped_QAM=zeros(counter,N_bits);
BER_QAM_theoritical=zeros(1,counter);
BER_QAM=zeros(1,counter);
for i= 1:counter
variance=sqrt((No(i)/2)*(E_avg_QAM/4));
real_noise=randn(1 , N_bits/4);
imag_noise=randn(1 , N_bits/4);
Noise_QAM=real_noise+1i*imag_noise;
Noise_QAM=variance*Noise_QAM;
QAM_noise(i,:)=QAM+Noise_QAM;
for j=1:N_bits/4
[value,index2]=min(abs(QAM_noise(i,j)-QAM_constellation));
Demapped_QAM(i,j)=index2-1;
Binary_value2=dec2bin(Demapped_QAM(i,j),4);
Binary_Demapped_QAM(i,(4*j-3):(4*j))=str2num(Binary_value2(:));
end
for k=1:N_bits
if data_bits(k) ==1 && Binary_Demapped_QAM(i,k)==0 || data_bits(k) ==0 && Binary_Demapped_QAM(i,k)==1
BER_QAM(1,i)=BER_QAM(1,i)+1;
end
end
BER_QAM_theoritical(i)=0.375*erfc(sqrt(1/(2.5*No(i))));
end
BER_QAM= BER_QAM/(N_bits);
figure('Name','BER')
semilogy(SNR_dB,BER_QAM,'g','Linewidth',3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
hold on
semilogy(SNR_dB,BER_QAM_theoritical,'b--','Linewidth',3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
legend( 'BER QAM','BER QAM theoritical','Location', 'best');
%% 8PSK
A=sqrt(1/2);
PSK_constellation = [1 A+1i*A -A+1i*A 1i A-1i*A -1i -1 -A-1i*A]; % 8PSK constellation points
reshaped_data4 = reshape(data_bits, 3, []);
reshaped_data4 = reshaped_data4';
decimal_number4 = bin2dec(num2str(reshaped_data4));
decimal_number4 = decimal_number4';
% Mapper
PSK = zeros(1, N_bits/3);
for i = 1:N_bits/3
PSK(i) = PSK_constellation(decimal_number4(i)+1);
end
E = abs(PSK).^2;
E_avg_PSK = sum(E)/(N_bits/3);
% Channel effect
PSK_noise = zeros(counter, N_bits/3);
Demapped_PSK = zeros(counter, N_bits/3);
Binary_Demapped_PSK = zeros(counter, N_bits);
BER_PSK_theoretical = zeros(1, counter);
BER_PSK = zeros(1, counter);
for i = 1:counter
variance = sqrt(No(i)/2*(E_avg_PSK/3));
real_noise=randn(1 , N_bits/3);
imag_noise=randn(1 , N_bits/3);
Noise_PSK=real_noise+1i*imag_noise;
Noise_PSK= variance*Noise_PSK;
PSK_noise(i, :) = PSK+Noise_PSK;
for j = 1:N_bits/3
[value, index] = min(abs(PSK_noise(i, j) - PSK_constellation));
Demapped_PSK(i, j) = index - 1;
Binary_value = dec2bin(Demapped_PSK(i, j), 3);
Binary_Demapped_PSK(i, (3*j-2):(3*j)) = str2num(Binary_value(:))';
end
for k = 1:N_bits
if data_bits(k) == 1 && Binary_Demapped_PSK(i, k) == 0 || data_bits(k) == 0 && Binary_Demapped_PSK(i, k) == 1
BER_PSK(1, i) = BER_PSK(1, i) + 1;
end
end
BER_PSK_theoretical(i) = erfc(sqrt(3/(No(i)))*sin(pi/8))/3;
end
BER_PSK = BER_PSK/N_bits;
figure('Name', 'BER')
semilogy(SNR_dB, BER_PSK, 'r', 'Linewidth', 3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
hold on;
semilogy(SNR_dB, BER_PSK_theoretical, 'b--', 'Linewidth', 3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
legend('BER 8PSK', 'BER 8PSK theoretical', 'Location', 'best');
%% BFSK
BFSK_constellation = [1i 1]; % BFSK constellation points
% Mapper
BFSK = zeros(1, N_bits);
for i = 1:N_bits
BFSK(i) = BFSK_constellation(data_bits(i)+1);
end
E = abs(BFSK).^2;
E_avg_BFSK = sum(E)/(N_bits);
% Channel effect
BFSK_noise = zeros(counter, N_bits);
Demapped_BFSK = zeros(counter, N_bits);
BER_BFSK_theoretical = zeros(1, counter);
BER_BFSK = zeros(1, counter);
for i = 1:counter
variance = sqrt(No(i)/2*E_avg_BFSK);
real_noise=randn(1 , N_bits);
imag_noise=randn(1 , N_bits);
Noise_BFSK=real_noise+1i*imag_noise;
Noise_BFSK= variance*Noise_BFSK;
BFSK_noise(i, :) = BFSK+Noise_BFSK;
%Demmaper
for j = 1:N_bits
[value, index] = min(abs(BFSK_noise(i, j) - BFSK_constellation));
Demapped_BFSK(i, j) = index - 1;
end
for k = 1:N_bits
if data_bits(k) == 1 && Demapped_BFSK(i, k) == 0 || data_bits(k) == 0 &&Demapped_BFSK(i, k) == 1
BER_BFSK(1, i) = BER_BFSK(1, i) + 1;
end
end
BER_BFSK_theoretical(i) = 0.5*erfc(sqrt(1/(2*No(i))));
end
BER_BFSK = BER_BFSK/N_bits;
figure('Name', 'BER')
semilogy(SNR_dB, BER_BFSK, 'r', 'Linewidth', 3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
hold on;
semilogy(SNR_dB, BER_BFSK_theoretical, 'b--', 'Linewidth', 3);
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
legend('BER BFSK', 'BER BFSK theoretical', 'Location', 'best');
%% Plot
figure('Name','BER')
semilogy(SNR_dB,BER_BPSK,'r','Linewidth',2);
hold on;
semilogy(SNR_dB,BER_BPSK_theoritical,'r--','Linewidth',2);
hold on;
semilogy(SNR_dB,BER_QPSK_gray,'y','Linewidth',2);
hold on;
semilogy(SNR_dB,BER_QPSK_gray_theoritical,'y--','Linewidth',2);
hold on;
semilogy(SNR_dB,BER_QPSK_nogray,'g','Linewidth',2);
hold on;
semilogy(SNR_dB,BER_QPSK_nogray_theoritical,'g--','Linewidth',2);
hold on;
semilogy(SNR_dB,BER_QAM,'b','Linewidth',2);
hold on;
semilogy(SNR_dB,BER_QAM_theoritical,'b--','Linewidth',2);
hold on;
semilogy(SNR_dB,BER_PSK,'c','Linewidth',2);
hold on;
semilogy(SNR_dB,BER_PSK_theoretical,'c--','Linewidth',2);
hold off;
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
legend('BER BPSK', 'BER BPSK theoritical','BER QPSK gray', 'BER QPSK gray theoritical','BER QPSK nogray', 'BER QPSK nogray theoritical','BER 16QAM', 'BER 16QAM theoretical','BER 8PSK', 'BER 8PSK theoretical', 'Location', 'best');
figure
semilogy(SNR_dB,BER_QPSK_gray,'g','Linewidth',2);
hold on;
semilogy(SNR_dB,BER_QPSK_nogray,'b','Linewidth',2);
hold off;
xlabel('Eb/No (dB)');
ylabel('BER');
ylim([1e-4, 1e0]);
grid on;
legend('BER QPSK gray','BER QPSK nogray','Location', 'best');
%% parameters
Tb=0.07;
Ts=0.01; % as used in project1
N_samples=7;
N_realizations=20000;
Eb=1;
t=Ts:Ts:Tb;
%% Bits generating and mapping
Data = randi ([0,1], N_realizations, 101);
Data_samples=repelem(Data, 1, N_samples);
BFSK_BBE=zeros(N_realizations,101*N_samples);
for i=1:N_realizations
for j=1:size(BFSK_BBE,2)
if(Data_samples(i,j)==0)
BFSK_BBE(i,j)=sqrt(Eb*2/Tb);
else
k = mod(j-1, N_samples)+1;
BFSK_BBE(i,j)=(cos(2*pi*t(k)*1/Tb)+1i*sin(2*pi*t(1,k)*1/Tb))*sqrt(Eb*2/Tb);
end
end
end
%% Randomization of initial time shift
delay = randi([0 6],N_realizations,1); % generation of random delay
BFSK_BB_delayed = zeros(N_realizations,700);
for i = 1:N_realizations % take 100 bits (700 samples) with random initial start and random bits
BFSK_BB_delayed(i,:) = BFSK_BBE(i, 1+delay(i) : 700+delay(i));
end
%% Autocorrelation
BFSK_BB_autocorrelation = zeros(1,700);
i=350;
for n = -349:350
for j= 1:N_realizations
BFSK_BB_autocorrelation(1,n+i) = BFSK_BB_autocorrelation(1,n+i)+(conj(BFSK_BB_delayed(j,i))*BFSK_BB_delayed(j,n+i));
end
end
BFSK_BB_autocorrelation= BFSK_BB_autocorrelation/N_realizations;
figure('Name','Statistical Auto correlation of BFSK BB');
i=1;
time = linspace(-25,25,700);
plot(time,abs(BFSK_BB_autocorrelation(i,:)));
title('Statistical Auto correlation BFSK Base Band');
xlabel('time');
ylabel('RX');
%% PSD
Fs = 1/Ts;
length = 700;
BFSK_BB_fft = fft(BFSK_BB_autocorrelation(1,:))/Fs; %to get power spectral density (PSD)
figure
f1=(-length/2:length/2-1)*(Tb)*Fs/length; % here to make the message centerd at zero
PSD_Practical=abs(fftshift(BFSK_BB_fft));
plot(f1,PSD_Practical);
title('BFSK Base Band PSD');
xlabel('frequency HZ');
ylabel('PSD');
ylim([0,2]);
%% Theoritical PSD
delta=zeros(1,size(f1,2));
tolerance = 1e-10;
index_half = find(abs(f1 - 0.5) < tolerance);
index_negative_half = find(abs(f1 + 0.5) < tolerance);
delta(1,index_half)=2/Tb;
delta(1,index_negative_half)=2/Tb;
f=(-length/2:length/2-1)*Fs/length;
PSD_Theoritical= delta+ (8 * cos(pi * Tb * f).^2 ) ./ (pi^2 * (4 * Tb^2 * f.^2 -1).^2);
figure
plot(f1,PSD_Theoritical);
title(' Theoritical PSD');
xlabel('frequency HZ');
ylabel('PSD');
ylim([0,2]);
figure
plot(f1,PSD_Practical,'b');
hold on
plot(f1,PSD_Theoritical,'r');
xlabel('frequency HZ');
ylabel('PSD');
ylim([0,2]);
legend('PSD Practical','PSD Theoritical','Location', 'best');