clc
clear all
%% 1
a1=[0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
0.45
0.5
0.6
];

T1=[116.125
110.033
118.695
112.412
109.607
113.915
111.881
114.409
111.718
108.467
107.248
];
AC1=[93.7
94.6
95
95.3
94.9
94.6
94
92.8
92.5
92.7
91.7
];
figure(1)
plot (a1,T1,'-^','MarkerSize',5,'MarkerFaceColor','b','MarkerEdgeColor','b','Color','b')
xlabel('Step Size')
ylabel('Computing Time(s)')
% title('Q3')
figure(2)
plot (a1,AC1,'-^','MarkerSize',5,'MarkerFaceColor','b','MarkerEdgeColor','b','Color','b')
% legend('GD','AGD','SGD');
xlabel('Step Size')
ylabel('Accuracy(%)')
%% 2
epoch=[1
2
3
4
5
6
7
8
9
10
15
20

];

T2=[23.763
44.528
69.812
92.381
116.311
134.838
157.42
179.107
200.053
227.186
341.173
535.16

];
AC2=[91.8
93.3
94.3
94.2
94.7
95.1
95.1
95.3
96
95.6
95.5
95.6

];
figure(3)
plot (epoch,T2,'-^','MarkerSize',5,'MarkerFaceColor','b','MarkerEdgeColor','b','Color','b')
% legend('GD','AGD','SGD');
xlabel('Epoch Number')
ylabel('Computing Time(s)')
% title('Q3')
figure(4)
plot (epoch,AC2,'-^','MarkerSize',5,'MarkerFaceColor','b','MarkerEdgeColor','b','Color','b')
% legend('GD','AGD','SGD');
xlabel('Epoch Number')
ylabel('Accuracy(%)')
%% 3
HID=[10
30
50
100
150
200
250
300
350
400
500
800

];

T3=[23.682
31.724
38.547
46.001
55.507
198.962
231.524
268.193
316.436
434.345
444.03
717.456


];
AC3=[84.3
92
92.5
94.9
95.1
95.5
95.4
95.3
95.2
95.6
95.9
95.8

];
figure(5)
plot (HID,T3,'-^','MarkerSize',5,'MarkerFaceColor','b','MarkerEdgeColor','b','Color','b')
% legend('GD','AGD','SGD');
xlabel('Hidden Node Number')
ylabel('Computing Time(s)')
% title('Q3')
figure(6)
plot (HID,AC3,'-^','MarkerSize',5,'MarkerFaceColor','b','MarkerEdgeColor','b','Color','b')
% legend('GD','AGD','SGD');
xlabel('Hidden Node Number')
ylabel('Accuracy(%)')
%% 4
lambda=[1.00E+02
1.00E+01
1.00E+00
1.00E-01
1.00E-02
1.00E-03
];
T4=[0.378
0.403
0.416
0.520
0.392
0.428
];
AC4=[80.5
80.8
80.9
80.3
80
79.7
];
figure(7)
plot (lambda,T4,'-^','MarkerSize',5,'MarkerFaceColor','b','MarkerEdgeColor','b','Color','b')
% legend('GD','AGD','SGD');
xlabel('Lambda')
ylabel('Computing Time(s)')
% title('Q3')
figure(8)
plot (lambda,AC4,'-^','MarkerSize',5,'MarkerFaceColor','b','MarkerEdgeColor','b','Color','b')
% legend('GD','AGD','SGD');
xlabel('Lambda')
ylabel('Accuracy(%)')
%% 5
