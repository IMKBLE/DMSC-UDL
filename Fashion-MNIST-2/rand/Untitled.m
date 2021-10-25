clc;
clear;

% %% cub
% load('EYB1.mat');
% cluser_num = 38;
% truth = groundtruth;
% %% multi-view
% addpath('E:\DAMC\wqq\MISSview\multiview');
% C{1} = X{1}';
% [reSC1,S,L] =Spectral_Clustering_cal101(C,cluser_num,truth',1);

addpath('E:\李朝阳研一科研\lizhaoyang\CLR_code');
load('EYB1.mat');
[result,S] = test(X{1,1}',groundtruth,38);
