close all
clc;
clear;
%% Requirement1
Data = randi ([0,1], 1, 10); % generate random data
Polar_NRZ=((2*Data)-1);      % Convert to 1 and -1
p=[5 4 3 2 1]/sqrt(55);      % Normalize the power of pulse shaping (matched filter).
Polar_NRZ_arr=upsample(Polar_NRZ,5); % Up sample the bits by 5 samples
figure('Name','T')
stem(Polar_NRZ_arr);
title('Data after upsampling');
xlabel('time');
ylabel('Amplitude');
 
y=conv(Polar_NRZ_arr,p); % Pulse shaping
y(end+1) = 0; % to match the size of the transmitted signal
 
figure('Name','Pulse Shaping');
plot(p,'r');
xlabel('time');
ylabel('Amplitude');
 
figure('Name','Tx output');
plot(y,'b');
xlabel('time');
ylabel('Amplitude');
 
matched_filter=fliplr(p); % generate matched filter at reciver h(t) = p(Ts-t)
figure('Name','Matched filter');
plot(matched_filter,'b');
xlabel('time');
ylabel('Amplitude');
 
y_padded = [zeros(1, length(matched_filter)-1), y]; 
 
y_rx_matched= conv(y_padded,matched_filter,'valid'); % output of matched filter at Rx
%paddign the transmitted signal then use valid in the convolution function 
%to generate output of the same size as the transmitted signal (55 points) 
 
sampling_arr = repmat([0 0 0 0 1],1,floor(size(y_rx_matched,2)/5));
%making sampling_arr to sample any output at the fifth sample (at Ts)
 
y_rx_matched_sampled = y_rx_matched.*sampling_arr; %to sample every Ts
 
filter_2=ones(1,5); % generate a square pulse for the second filter
Eb_filter_2= sum(filter_2(:).^2); %get the energy of the filter
filter_2=filter_2/sqrt(Eb_filter_2); % Normalize the second filter
y_rx_unmatched= conv(y_padded,filter_2,'valid');    % Output of unmatched filter at Rx
y_rx_unmatched_sampled = y_rx_unmatched.*sampling_arr; %to sample every Ts
 
figure('Name','Unmatched filter');
plot(filter_2,'b');
xlabel('Time [Ts sec]');
ylabel('Amplitude');
points = linspace(0.2, 11, 55); %to make 55 points starting from 0.2 to 11
 
figure('Name','Output of Two filters')
subplot(2,1,1)
plot(points,y_rx_matched,'g','Linewidth',3);
xlabel('Time [Ts sec]');
ylabel('Amplitude');
title('output of matched filter');
hold on;           % to plot on the same plot
stem(points,y_rx_matched_sampled,'r','Linewidth',3);
xlabel('Time [Ts sec]');
ylabel('Amplitude');
legend('out of matched filter', 'out of matched filter sampled', 'Location', 'best');
xlim([0, 11]);
hold off;
 
 
 
subplot(2,1,2)
plot(points,y_rx_unmatched,'b','Linewidth',3);
xlabel('Time [Ts sec]');
ylabel('Amplitude');
title('output of unmatched filter');
hold on;
stem(points,y_rx_unmatched_sampled,'r','Linewidth',3);
xlabel('Time [Ts sec]');
ylabel('Amplitude');
legend('out of unmatched filter', 'out of unmatched filter sampled', 'Location', 'best');
xlim([0, 11]);
hold off;
 
 
% Use correlator
correlator_out=zeros(1,55);
 
for i = 1:5:size(y,2) 
    multiplication =  y(i:i+4) .* p;
    for j =1:5
       correlator_out(1,i+j-1) =  sum(multiplication(1,1:j));
    end
end
% multiply the transmitted signal then integrate and dump
 
figure('Name','Output of matched filter and correlator')
plot(points,y_rx_matched,'g','Linewidth',3);
xlabel('Time [Ts sec]');
ylabel('Amplitude');
hold on
plot(points,correlator_out,'b','Linewidth',3);
xlabel('Time [Ts sec]');
ylabel('Amplitude');
hold off;
xlim([0, 11]);
legend('out of matched filter', 'out of correlator', 'Location', 'best');
 
 
correlator_out_sampled=correlator_out.*sampling_arr;
 
figure('Name','Output of sampled correlator and correlator')
plot(points,correlator_out,'g','Linewidth',3);
xlabel('Time [Ts sec]');
ylabel('Amplitude');
hold on
stem(points,correlator_out_sampled,'b','Linewidth',3);
xlabel('Time [Ts sec]');
ylabel('Amplitude');
hold off;
xlim([0, 11]);
legend('out of sampled correlator', 'out of  correlator', 'Location', 'best');
 
 
%% Noise Analysis
Data = randi ([0,1], 1, 10000); % generate new random data with 10000
Polar_NRZ=((2*Data)-1);         % Convert to 1 and -1
p=[5 4 3 2 1]/sqrt(55);         % normalize the power of the pulse
Polar_NRZ_arr=upsample(Polar_NRZ,5); % Up sample the bits by 5 samples
y=conv(p,Polar_NRZ_arr);
y(end+1) = 0;
 
 
No=zeros(1,8); % an array contian the power of Noise.
SNR=zeros(1,8); % an array contian the linear values of Eb/No.
BER=zeros(1,8);% an array contian the values of theoretical BER .
Prob_of_error_matched=zeros(1,8);% an array contian the values of BER when using matched filter. .
Prob_of_error_unmatched=zeros(1,8);% an array contian the values of BER when using unmatched filter. 
for snr_db= -2:1:5
    SNR(1,snr_db+3)=10^(snr_db/10); % Calculate linear values.
    No(1,snr_db+3)=1/SNR(1,snr_db+3); % Calculate No.
end
noise=randn(1,size(y,2)); % generate random noise
for i=1:8
    Noise=sqrt(No(1,i)/2)*noise; % multiply by variance.
    v=y+Noise;                   % Add the noise at Rx
    
    v_padded = [zeros(1, length(matched_filter)-1), v];
    
    v_rx_matched_noise= conv(v_padded,matched_filter,'valid'); % The output of matched filter
    v_rx_unmatched_noise= conv(v_padded,filter_2,'valid');     % The output of unmatched filter
    %Calculate the probability of error.
    for j=1:10000
        if Data(1,j) ==1 && v_rx_matched_noise(1,j*5)<0 || Data(1,j) ==0 && v_rx_matched_noise(1,j*5)>0 % check if the output of the matched filter less than zero (which mapped to 0) and the input was 1 or the output greater than zero (which mapped to 1) and input was 0
            Prob_of_error_matched(1,i)=Prob_of_error_matched(1,i)+1;
        end
        if Data(1,j) ==1 && v_rx_unmatched_noise(1,j*5)<0 || Data(1,j) ==0 && v_rx_unmatched_noise(1,j*5)>0 % check if the output of the unmatched filter less than zero (which mapped to 0) and the input was 1 or the output greater than zero (which mapped to 1) and input was 0
            Prob_of_error_unmatched(1,i)=Prob_of_error_unmatched(1,i)+1;
        end
    end
    BER(1,i)=0.5*erfc(sqrt(SNR(1,i))); % calculate the theoritical BER
    Prob_of_error_matched(1,i)=Prob_of_error_matched(1,i)/10000;
    Prob_of_error_unmatched(1,i)=Prob_of_error_unmatched(1,i)/10000;
end
x=-2:1:5; % Eb/No in dB
figure('Name','BER')
semilogy(x,BER(1,:),'g','Linewidth',3); %plot BER using logscale
xlabel('Eb/No (dB)');
ylabel('BER');
 
hold on
semilogy(x,Prob_of_error_matched(1,:),'b','Linewidth',3);
xlabel('Eb/No (dB)');
ylabel('BER');
 
hold on
semilogy(x,Prob_of_error_unmatched(1,:),'r','Linewidth',3);
xlabel('Eb/No (dB)');
ylabel('BER');
legend('BER for erfc', 'BER for matched filter',  'BER for unmatched filter','Location', 'best');
%% Eye diagram
R=[0,0,1,1];
DELAY=[2,8,2,8];
Data_3 = randi ([0,1], 1, 100);
Polar_NRZ=((2*Data_3)-1);
Polar_NRZ_arr=upsample(Polar_NRZ,5);
 
Y_TX=cell(1,4); % making array of arrays each of them contaning the result of the convolution 
Y_TX_padded=cell(1,4);
Y_RX=cell(1,4);
for i=1:4
rcosine_filter = rcosine(1 , 5 , 'sqrt' , R(1,i), DELAY(1,i));
Y_TX{i}=conv(Polar_NRZ_arr,rcosine_filter);
Y_TX_padded{i} = [zeros(1, length(rcosine_filter)-1), Y_TX{i}];
Y_RX{i}=conv(Y_TX_padded{i},rcosine_filter,'valid');
eye_fig = eyediagram(  Y_TX{i} , 10);
title(['Eye diagram at A for R= ', num2str(R(1,i)), ' and Delay= ', num2str(DELAY(1,i))]);
eye_fig = eyediagram(  Y_RX{i} , 10);
title(['Eye diagram at B for R= ', num2str(R(1,i)), ' and Delay= ', num2str(DELAY(1,i))]);
 
end
