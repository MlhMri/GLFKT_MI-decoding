
% This script is an implementation of the method proposed in:

% Miri, M., et al., "Spectral representation of EEG data using learned graphs
% with application to motor imagery decoding." Biomedical Signal Processing
% and Control 87 (2024): 105537.
% https://www.sciencedirect.com/science/article/pii/S1746809423009709


% The general idea is to transform EEG data into a spectral representation.
% First, subject-specific EEG graphs are inferred. By interpreting EEG maps
% as time-dependent graph signals on the derived graphs, the data is
% transformed into a spectral representation. Then, a discriminative spectral
% graph subspace is derived that specifically aims at differentiating
% two-class data, using a projection matrix obtained by the Fukunaga–Koontz
% transform (FKT); the transform takes in to account differences in temporal
% covariation of spectral profiles in the two classes. The logarithm of
% variance of representations within the subspace is treated as features,
% which is in turn used to train and test an SVM classifier.


% Maliheh Miri, December 2024.
%=========================================================================%


clear
close all

subject = ['a','l','v','w','y'];
whichgraph = 'log-penalized';   % 'correlation','log-penalized','l2-penalized'
Tot_Acc = [];
addpath(genpath('utils'));

for sbj = 1:5

    load(['data/data_set_IVa_a',subject(sbj),'.mat']);
    load(['data/true_labels_a',subject(sbj),'.mat']);

    %-------------1. bandpass filtering & extracting trials---------------%

    cnt = double(cnt)*0.1; % micro volt
    Fs = 100;
    [b,a] = butter(3,[8 30]/(Fs/2),'bandpass');
    cnt = filtfilt(b,a,cnt);

    group = mrk.y;
    pos = mrk.pos;
    Tot_trial = [];
    c1 = 0;
    c2 = 0;
    traindata1 = [];
    traindata2 = [];
    for i = 1:length(group)
        ind = pos(i)+(0.5*100)-1:pos(i)+(2.5*100)-1;  % 0.5-2.5 seconds
        trial = cnt(ind,:);
        if group(i) == 1
            c1 = c1+1;
            traindata1(:,:,c1) = trial;
        elseif group(i) == 2
            c2 = c2+1;
            traindata2(:,:,c2) = trial;
        end
    end

    for i = 1:(c1+c2)
        ind = pos(i)+(0.5*100)-1:pos(i)+(2.5*100)-1;
        trial = cnt(ind,:);
        Tot_trial = [Tot_trial;trial];
    end
    signal = Tot_trial';

    testdata1 = [];
    testdata2 = [];
    c1t = 0;
    c2t = 0;
    for i = test_idx(1):test_idx(end)
        ind = pos(i)+(0.5*100)-1:pos(i)+(2.5*100)-1;
        trial = cnt(ind,:);
        if true_y(i) == 1
            c1t = c1t+1;
            testdata1(:,:,c1t) = trial;
        elseif true_y(i) == 2
            c2t = c2t+1;
            testdata2(:,:,c2t) = trial;
        end
    end

    %----------------------2. deriving brain graphs-----------------------%

    switch whichgraph

        case 'correlation'
            GSize = size(cnt,2);    % graph size
            A = zeros(GSize,GSize);
            gsignal = signal';
            Corr = [];
            for p = 1:1:GSize
                for q = 1:1:GSize
                    cor(q) = corr(gsignal(:,p),gsignal(:,q));
                end
                Corr = [Corr;cor];
            end
            A = abs(Corr);
            A = A.*~eye(size(A));
            A = (A+A')/2;

        case 'log-penalized'
            addpath(genpath('gspbox'))
            Dis = pdist(signal,'euclidean');
            Z = squareform(Dis);
            Z = Z./max(Z(:));
            alpha_k = 0.6; beta_k = 0.5;
            A = gsp_learn_graph_log_degrees(Z, alpha_k, beta_k);
            A(A<1e-4) = 0;

        case 'l2-penalized'
            addpath(genpath('gspbox'))
            Dis = pdist(signal,'euclidean');
            Z = squareform(Dis);
            Z = Z./max(Z(:));
            alpha_k = 0.6;
            A = gsp_learn_graph_l2_degrees(Z, alpha_k);
            A(A<1e-4) = 0;
    end

    %----------3. eigen-decomposition of the graph's Laplacian------------%

    addpath(genpath('sgwt_toolbox'));
    L = sgwt_laplacian(A,'opt','normalized');   % normalized Laplacian
    [U,D] = eig(full(L));
    E = diag(D);
    [~,dd] = sort(E);
    E = E(dd);        % graph eigenvalues
    U = U(:,dd);      % graph eigenvectors

    % graph's frequency subbands:
    UL = U(:,1:floor(size(U,2)/3));
    UM = U(:,floor((size(U,2)/3)+1):floor((2*size(U,2))/3));
    UH = U(:,floor(((2*size(U,2))/3)+1):end);

    %----------------4. demean & normalize graph signals------------------%

    traindata1_norm = demean_norm (traindata1, U);
    traindata2_norm = demean_norm (traindata2, U);
    testdata1_norm = demean_norm (testdata1, U);
    testdata2_norm = demean_norm (testdata2, U);

    %------------------------5. GFT computation---------------------------%

    U_fc = UL;    % LF sub-band
    traindata1_gft=[];
    traindata2_gft=[];
    for i1=1:size(traindata1,3)
        gft1 = (U_fc'*traindata1_norm(:,:,i1))';
        traindata1_gft(:,:,i1) = gft1(:,2:end);
    end
    for i2=1:size(traindata2,3)
        gft2 = (U_fc'*traindata2_norm(:,:,i2))';
        traindata2_gft(:,:,i2) = gft2(:,2:end);
    end

    %------------------------------6. FKT---------------------------------%

    W = fkt(traindata1_gft,traindata2_gft,1);

    %---------------7. feature extraction & classification----------------%

    Featuretrain1=[];Featuretrain2=[];Featuretest1=[];Featuretest2=[];
    N1 = size(traindata1,3);
    N2 = size(traindata2,3);
    mx = max([N1 N2]);
    for k = 1:mx
        if k <= N1
            x1 = traindata1_norm(:,:,k);
            x1_gft = U_fc'*x1;
            x1 = x1_gft(2:end,:);
            x1 = W*x1;
            Featuretrain1(k,:)= log(var(x1'));
        end
        if k <= N2
            x2 = traindata2_norm(:,:,k);
            x2_gft = U_fc'*x2;
            x2 = x2_gft(2:end,:);
            x2 = W*x2;
            Featuretrain2(k,:) = log(var(x2'));

        end
    end
    dtrain = [ones(1,size(Featuretrain1,1)),-ones(1,size(Featuretrain2,1))];
    datatrain = [Featuretrain1;Featuretrain2]';

    SVMStruct = fitcsvm(datatrain', dtrain,'KernelFunction','linear','Standardize','on','BoxConstraint',1000);

    N1 = size(testdata1,3);
    N2 = size(testdata2,3);
    mxt = max([N1,N2]);
    for k = 1:mxt
        if k <= N1
            x1 = testdata1_norm(:,:,k);
            x1_gft = U_fc'*x1;
            x1 = x1_gft(2:end,:);
            x1 = W*x1;
            Featuretest1(k,:) = log(var(x1'));
        end
        if k <= N2
            x2 = testdata2_norm(:,:,k);
            x2_gft = U_fc'*x2;
            x2 = x2_gft(2:end,:);
            x2 = W*x2;
            Featuretest2(k,:) = log(var(x2'));

        end
    end
    datatest = [Featuretest1;Featuretest2]';
    dtest = [ones(1,size(Featuretest1,1)),-ones(1,size(Featuretest2,1))];

    output = predict(SVMStruct,datatest');

    C = confusionmat(dtest,output);
    Totalaccuracy=  sum(diag(C)) / sum(C(:)) *100;
    Tot_Acc = [Tot_Acc, Totalaccuracy]
    mean(Tot_Acc);
    std(Tot_Acc);
end

disp(['Total Accuracy: ',num2str(mean(Tot_Acc)) , ' ± ', num2str(std(Tot_Acc)) ,' %']);
