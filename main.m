clear all % clear all variables from memory
close all % close all open figures

%-------------------------------------------------------------------------

% Global Parameters
figureNumber = 1; %used to open multiple figures, 1 for each 
dotColors = [0 0 1; 1 0 0; 0 153/255 76/255];
lineColor = [0 0 0; 143/255 93/255 179/255; 227/255 180/255 27/255];
case1ColourScheme = [
    204/255, 229/255, 1
    1, 204/255, 204/255];
case2ColourScheme = [
    204/255, 229/255, 1
    1, 204/255, 204/258
    204/255, 1, 204/255];

%-------------------------------------------------------------------------

% Case 1 Classes
A = classCreator(1, 200, [5 10], [8 0; 0 4], 400);
B = classCreator(1, 200, [10 15], [8 0; 0 4], 400);
% Graphing Case 1 Feature Space
figure(figureNumber);
scatter(A.cluster(:,1), A.cluster(:,2), 20, dotColors(1,:),'filled'); hold on;
scatter(B.cluster(:,1), B.cluster(:,2), 20, dotColors(2,:),'filled'); hold on;
plot(A.stdContour(:,1), A.stdContour(:,2), 'LineWidth', 2,'color',dotColors(1,:)); hold on;
plot(B.stdContour(:,1), B.stdContour(:,2), 'LineWidth', 2,'color',dotColors(2,:)); hold on;
xlabel('Feature 1');
ylabel('Feature 2');
title('Case 1 - Feature Space');
figureNumber = figureNumber + 1;

% Case 2 Classes
C = classCreator(2, 100, [5 10], [8 4; 4 40], 450);
D = classCreator(2, 200, [15 10], [8 0; 0 8], 450);
E = classCreator(2, 150, [10 5], [10 -5; -5 20], 450);
% Graphing Case 2 Feature Space
figure(figureNumber);
scatter(C.cluster(:,1), C.cluster(:,2), 20, dotColors(1,:),'filled'); hold on;
scatter(D.cluster(:,1), D.cluster(:,2), 20, dotColors(2,:),'filled'); hold on;
scatter(E.cluster(:,1), E.cluster(:,2), 20, dotColors(3,:),'filled'); hold on;
plot(C.stdContour(:,1), C.stdContour(:,2), 'LineWidth', 2,'color',dotColors(1,:)); hold on;
plot(D.stdContour(:,1), D.stdContour(:,2), 'LineWidth', 2,'color',dotColors(2,:)); hold on;
plot(E.stdContour(:,1), E.stdContour(:,2), 'LineWidth', 2,'color',dotColors(3,:)); hold on;
xlabel('Feature 1');
ylabel('Feature 2');
title('Case 2 - Feature Space');
figureNumber = figureNumber + 1;

%-------------------------------------------------------------------------

% CASE 1
% Generate the grid
case1GRID = gridMaker(min([min(A.cluster(:,1)),min(B.cluster(:,1))])-1, max([max(A.cluster(:,1)),max(B.cluster(:,1))])+1, min([min(A.cluster(:,2)),min(B.cluster(:,2))])-1, max([max(A.cluster(:,2)),max(B.cluster(:,2))])+1);

% Create classification matrices for the different classifiers and compute
[case1GRID.grid(:,:,3),case1GRID.grid(:,:,4),case1GRID.grid(:,:,5), case1GRID.grid(:,:,6), case1GRID.grid(:,:,7)] = allFunctions.classification(case1GRID.grid, [A; B]);

% Plot the 3 classifier decision boundaries
figureNumber = allFunctions.plotterFull(case1GRID.grid, 3, [A; B],'MED: Case 1', case1ColourScheme, figureNumber); % Case 1 MED Plot
figureNumber = allFunctions.plotterFull(case1GRID.grid, 4, [A; B],'GED: Case 1', case1ColourScheme, figureNumber); % Case 1 GED Plot
figureNumber = allFunctions.plotterFull(case1GRID.grid, 5, [A; B],'MAP: Case 1', case1ColourScheme, figureNumber); % Case 1 MAP Plot
% Graphing the MED/GED/MAP decision boundaries on the same Plot
figure(figureNumber);
contour(case1GRID.grid(:,:,1), case1GRID.grid(:,:,2), case1GRID.grid(:,:,3), 'color', lineColor(1,:)); hold on;
contour(case1GRID.grid(:,:,1), case1GRID.grid(:,:,2), case1GRID.grid(:,:,4), 'color', lineColor(2,:)); hold on;
contour(case1GRID.grid(:,:,1), case1GRID.grid(:,:,2), case1GRID.grid(:,:,5), 'color', lineColor(2,:)); hold on;
scatter(A.cluster(:,1), A.cluster(:,2), 20, dotColors(1,:),'filled'); hold on;
scatter(B.cluster(:,1), B.cluster(:,2), 20, dotColors(2,:),'filled'); hold on;
plot(A.stdContour(:,1), A.stdContour(:,2), 'LineWidth', 2,'color',dotColors(1,:)); hold on;
plot(B.stdContour(:,1), B.stdContour(:,2), 'LineWidth', 2,'color',dotColors(2,:)); hold on;
xlabel('Feature 1');
ylabel('Feature 2');
title('Case 1 - MED, GED and MAP');
figureNumber = figureNumber + 1;

% Plot the 2 classifier decision boundaries
figureNumber = allFunctions.plotterFull(case1GRID.grid, 6, [A; B],'Nearest Neighbors: Case 1', case1ColourScheme, figureNumber); % Case 1 NN Plot
figureNumber = allFunctions.plotterFull(case1GRID.grid, 7, [A; B],'k-Nearest Neighbors: Case 1', case1ColourScheme, figureNumber); % Case 1 KNN Plot
% Graphing the NN/kNN decision boundaries on the same Plot
figure(figureNumber);
contour(case1GRID.grid(:,:,1), case1GRID.grid(:,:,2), case1GRID.grid(:,:,6), 'color', lineColor(1,:)); hold on;
contour(case1GRID.grid(:,:,1), case1GRID.grid(:,:,2), case1GRID.grid(:,:,7), 'color', lineColor(2,:)); hold on;
scatter(A.cluster(:,1), A.cluster(:,2), 20, dotColors(1,:),'filled'); hold on;
scatter(B.cluster(:,1), B.cluster(:,2), 20, dotColors(2,:),'filled'); hold on;
plot(A.stdContour(:,1), A.stdContour(:,2), 'LineWidth', 2,'color',dotColors(1,:)); hold on;
plot(B.stdContour(:,1), B.stdContour(:,2), 'LineWidth', 2,'color',dotColors(2,:)); hold on;
xlabel('Feature 1');
ylabel('Feature 2');
title('Case 1 - NN and kNN');
figureNumber = figureNumber + 1;

%Confusion Matrices for Case 1 classifiers
confMatrixMED1 = [errorFunctions.getErrorCount(A.cluster, 1, case1GRID, 3) errorFunctions.getErrorCount(A.cluster, 2, case1GRID, 3); errorFunctions.getErrorCount(B.cluster, 1, case1GRID, 3) errorFunctions.getErrorCount(B.cluster, 2, case1GRID, 3)];
confMatrixGED1 = [errorFunctions.getErrorCount(A.cluster, 1, case1GRID, 4) errorFunctions.getErrorCount(A.cluster, 2, case1GRID, 4); errorFunctions.getErrorCount(B.cluster, 1, case1GRID, 4) errorFunctions.getErrorCount(B.cluster, 2, case1GRID, 4)];
confMatrixMAP1 = [errorFunctions.getErrorCount(A.cluster, 1, case1GRID, 5) errorFunctions.getErrorCount(A.cluster, 2, case1GRID, 5); errorFunctions.getErrorCount(B.cluster, 1, case1GRID, 5) errorFunctions.getErrorCount(B.cluster, 2, case1GRID, 5)];
confMatrixNN1 = [errorFunctions.getErrorCount(A.testCluster, 1, case1GRID, 6) errorFunctions.getErrorCount(A.testCluster, 2, case1GRID, 6); errorFunctions.getErrorCount(B.testCluster, 1, case1GRID, 6) errorFunctions.getErrorCount(B.testCluster, 2, case1GRID, 6)];
confMatrixKNN1 = [errorFunctions.getErrorCount(A.testCluster, 1, case1GRID, 7) errorFunctions.getErrorCount(A.testCluster, 2, case1GRID, 7); errorFunctions.getErrorCount(B.testCluster, 1, case1GRID, 7) errorFunctions.getErrorCount(B.testCluster, 2, case1GRID, 7)];

%-------------------------------------------------------------------------

% CASE 2
% Generate the grid
case2GRID = gridMaker(min([min(C.cluster(:,1)),min(D.cluster(:,1)),min(E.cluster(:,1))])-1, max([max(C.cluster(:,1)),max(D.cluster(:,1)),max(E.cluster(:,1))])+1, min([min(C.cluster(:,2)),min(D.cluster(:,2)),min(E.cluster(:,2))])-1, max([max(C.cluster(:,2)),max(D.cluster(:,2)),max(E.cluster(:,2))])+1);

% Create classification matrices for the different classifiers and compute
[case2GRID.grid(:,:,3),case2GRID.grid(:,:,4),case2GRID.grid(:,:,5), case2GRID.grid(:,:,6), case2GRID.grid(:,:,7)]  = allFunctions.classification(case2GRID.grid, [C; D; E]);

% Plot the 3 classifier decision boundaries
figureNumber = allFunctions.plotterFull(case2GRID.grid, 3, [C; D; E],'MED: Case 2', case2ColourScheme, figureNumber); % Case 2 MED Plot
figureNumber = allFunctions.plotterFull(case2GRID.grid, 4, [C; D; E],'GED: Case 2', case2ColourScheme, figureNumber); % Case 2 GED Plot
figureNumber = allFunctions.plotterFull(case2GRID.grid, 5, [C; D; E],'MAP: Case 2', case2ColourScheme, figureNumber); % Case 2 MAP Plot
% Graphing the MED/GED/MAP decision boundaries on the same Plot
figure(figureNumber);
contour(case2GRID.grid(:,:,1), case2GRID.grid(:,:,2), case2GRID.grid(:,:,3), 'color', lineColor(1,:)); hold on;
contour(case2GRID.grid(:,:,1), case2GRID.grid(:,:,2), case2GRID.grid(:,:,4), 'color', lineColor(2,:)); hold on;
contour(case2GRID.grid(:,:,1), case2GRID.grid(:,:,2), case2GRID.grid(:,:,5), 'color', lineColor(3,:)); hold on;
scatter(C.cluster(:,1), C.cluster(:,2), 20, dotColors(1,:),'filled'); hold on;
scatter(D.cluster(:,1), D.cluster(:,2), 20, dotColors(2,:),'filled'); hold on;
scatter(E.cluster(:,1), E.cluster(:,2), 20, dotColors(3,:),'filled'); hold on;
plot(C.stdContour(:,1), C.stdContour(:,2), 'LineWidth', 2,'color',dotColors(1,:)); hold on;
plot(D.stdContour(:,1), D.stdContour(:,2), 'LineWidth', 2,'color',dotColors(2,:)); hold on;
plot(E.stdContour(:,1), E.stdContour(:,2), 'LineWidth', 2,'color',dotColors(3,:)); hold on;
xlabel('Feature 1');
ylabel('Feature 2');
title('Case 2 - MED, GED and MAP');
figureNumber = figureNumber + 1;

% Plot the 2 classifier decision boundaries
figureNumber = allFunctions.plotterFull(case2GRID.grid, 6, [C; D; E],'Nearest Neighbors: Case 2', case2ColourScheme, figureNumber); % Case 2 NN Plot
figureNumber = allFunctions.plotterFull(case2GRID.grid, 7, [C; D; E],'k-Nearest Neighbors: Case 2', case2ColourScheme, figureNumber); % Case 2 kNN Plot
%Graphing the NN/kNN decision boundaries on the same Plot
figure(figureNumber);
contour(case2GRID.grid(:,:,1), case2GRID.grid(:,:,2), case2GRID.grid(:,:,6), 'color', lineColor(1,:)); hold on;
contour(case2GRID.grid(:,:,1), case2GRID.grid(:,:,2), case2GRID.grid(:,:,7), 'color', lineColor(2,:)); hold on;
scatter(C.cluster(:,1), C.cluster(:,2), 20, dotColors(1,:),'filled'); hold on;
scatter(D.cluster(:,1), D.cluster(:,2), 20, dotColors(2,:),'filled'); hold on;
scatter(E.cluster(:,1), E.cluster(:,2), 20, dotColors(3,:),'filled'); hold on;
plot(C.stdContour(:,1), C.stdContour(:,2), 'LineWidth', 2,'color',dotColors(1,:)); hold on;
plot(D.stdContour(:,1), D.stdContour(:,2), 'LineWidth', 2,'color',dotColors(2,:)); hold on;
plot(E.stdContour(:,1), E.stdContour(:,2), 'LineWidth', 2,'color',dotColors(3,:)); hold on;
xlabel('Feature 1');
ylabel('Feature 2');
title('Case 2 - NN and kNN');
figureNumber = figureNumber + 1;

% Confusion Matrices for Case 2 classifiers
confMatrixMED2 = [errorFunctions.getErrorCount(C.cluster, 1, case2GRID, 3) errorFunctions.getErrorCount(C.cluster, 2, case2GRID, 3) errorFunctions.getErrorCount(C.cluster, 3, case2GRID, 3)
                  errorFunctions.getErrorCount(D.cluster, 1, case2GRID, 3) errorFunctions.getErrorCount(D.cluster, 2, case2GRID, 3) errorFunctions.getErrorCount(D.cluster, 3, case2GRID, 3)
                  errorFunctions.getErrorCount(E.cluster, 1, case2GRID, 3) errorFunctions.getErrorCount(E.cluster, 2, case2GRID, 3) errorFunctions.getErrorCount(E.cluster, 3, case2GRID, 3)];
              
confMatrixGED2 = [errorFunctions.getErrorCount(C.cluster, 1, case2GRID, 4) errorFunctions.getErrorCount(C.cluster, 2, case2GRID, 4) errorFunctions.getErrorCount(C.cluster, 3, case2GRID, 4)
                  errorFunctions.getErrorCount(D.cluster, 1, case2GRID, 4) errorFunctions.getErrorCount(D.cluster, 2, case2GRID, 4) errorFunctions.getErrorCount(D.cluster, 3, case2GRID, 4)
                  errorFunctions.getErrorCount(E.cluster, 1, case2GRID, 4) errorFunctions.getErrorCount(E.cluster, 2, case2GRID, 4) errorFunctions.getErrorCount(E.cluster, 3, case2GRID, 4)];
              
confMatrixMAP2 = [errorFunctions.getErrorCount(C.cluster, 1, case2GRID, 5) errorFunctions.getErrorCount(C.cluster, 2, case2GRID, 5) errorFunctions.getErrorCount(C.cluster, 3, case2GRID, 5)
                  errorFunctions.getErrorCount(D.cluster, 1, case2GRID, 5) errorFunctions.getErrorCount(D.cluster, 2, case2GRID, 5) errorFunctions.getErrorCount(D.cluster, 3, case2GRID, 5)
                  errorFunctions.getErrorCount(E.cluster, 1, case2GRID, 5) errorFunctions.getErrorCount(E.cluster, 2, case2GRID, 5) errorFunctions.getErrorCount(E.cluster, 3, case2GRID, 5)]; 
              
confMatrixNN2 = [errorFunctions.getErrorCount(C.testCluster, 1, case2GRID, 6) errorFunctions.getErrorCount(C.testCluster, 2, case2GRID, 6) errorFunctions.getErrorCount(C.testCluster, 3, case2GRID, 6)
                 errorFunctions.getErrorCount(D.testCluster, 1, case2GRID, 6) errorFunctions.getErrorCount(D.testCluster, 2, case2GRID, 6) errorFunctions.getErrorCount(D.testCluster, 3, case2GRID, 6)
                 errorFunctions.getErrorCount(E.testCluster, 1, case2GRID, 6) errorFunctions.getErrorCount(E.testCluster, 2, case2GRID, 6) errorFunctions.getErrorCount(E.testCluster, 3, case2GRID, 6)]; 
          
confMatrixKNN2 = [errorFunctions.getErrorCount(C.testCluster, 1, case2GRID, 7) errorFunctions.getErrorCount(C.testCluster, 2, case2GRID, 7) errorFunctions.getErrorCount(C.testCluster, 3, case2GRID, 7)
                  errorFunctions.getErrorCount(D.testCluster, 1, case2GRID, 7) errorFunctions.getErrorCount(D.testCluster, 2, case2GRID, 7) errorFunctions.getErrorCount(D.testCluster, 3, case2GRID, 7)
                  errorFunctions.getErrorCount(E.testCluster, 1, case2GRID, 7) errorFunctions.getErrorCount(E.testCluster, 2, case2GRID, 7) errorFunctions.getErrorCount(E.testCluster, 3, case2GRID, 7)]; 