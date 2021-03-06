classdef allFunctions
    methods(Static)
        function [medclassificationGrid, gedclassificationGrid, MAPclassificationGrid, NNclassificationGrid, kNNclassificationGrid] = classification(grid, classes)
            % output grids created in memory of the size of input
            medclassificationGrid = zeros(size(grid,1),size(grid,2));
            gedclassificationGrid = zeros(size(grid,1),size(grid,2));
            MAPclassificationGrid = zeros(size(grid,1),size(grid,2));
            NNclassificationGrid = zeros(size(grid,1),size(grid,2));
            kNNclassificationGrid = zeros(size(grid,1),size(grid,2));
            
            % creates arrays of the size of the number of classes
            MEDdistances2Means = size(classes,1);
            GEDdistances2Means = size(classes,1);
            MAPdistances2Means = size(classes,1);
            NNdistances2Means = size(classes,1);
            kNNdistances2Means = size(classes,1);
            
            % iterate through the input grid
            for i = 1:size(grid, 1)
                for j = 1:size(grid, 2)
                    for index = 1:size(classes,1)
                        % adds classification values for each class
                        MEDdistances2Means(index) = allFunctions.eDist(grid(i,j,1), grid(i,j,2), classes(index).mu);
                        GEDdistances2Means(index) = allFunctions.geDist(grid(i,j,1), grid(i,j,2), classes(index).mu, classes(index).sigma); 
                        MAPdistances2Means(index) = allFunctions.maxAPrior(grid(i,j,1), grid(i,j,2), classes(index).mu, classes(index).sigma, classes(index).prior);
                        NNdistances2Means(index) = allFunctions.find_NN(grid(i,j,1), grid(i,j,2), classes(index).cluster);
                        kNNdistances2Means(index) = allFunctions.find_KNN(grid(i,j,1), grid(i,j,2), classes(index).cluster);
                    end
                    % determines the classification and adds to output grid
                    [~, MEDclassification] = min(MEDdistances2Means); medclassificationGrid(i,j) = MEDclassification;
                    [~, GEDclassification] = min(GEDdistances2Means); gedclassificationGrid(i,j) = GEDclassification;
                    [~, MAPclassification] = max(MAPdistances2Means); MAPclassificationGrid(i,j) = MAPclassification;
                    [~, NNclassification] = min(NNdistances2Means); NNclassificationGrid(i,j) = NNclassification;
                    [~, kNNclassification] = min(kNNdistances2Means); kNNclassificationGrid(i,j) = kNNclassification;
                end
            end
        end

        function euclideanDistance = eDist(x,y,mu)
            % calculates euclidean distance
            euclideanDistance = round(sqrt((x-mu(1))^2 + (y-mu(2))^2), 3);
        end

        function geDistance = geDist(x,y,u,s)
            % calculates generalized euclidean distance
            u = transpose(u);
            X = [x; y];
            h = X - u;
            r = transpose(h);
            geDistance = sqrt((r)*inv(s)*(h));
        end

        function probMAP = maxAPrior(x, y, u, s, prior)
            % calculates MAP probability
            u = u';
            X = [x; y];
            h = X - u;
            r = h';
            probMAP = prior*(exp(-0.5*((r)*inv(s)*(h))))/(2*pi*sqrt(det(s)));
        end
        
        function NN = find_NN(x, y, cluster)
            % calculates distance to nearest neighbour
            %for loop to loop through every data point in each cluster
            minDist = sqrt((x-cluster(1,1))^2 + (y-cluster(1,2))^2);
            for i = 1:size(cluster, 1)
                dist = sqrt((x-cluster(i, 1))^2+(y-cluster(i, 2))^2);
                if(dist < minDist)
                    minDist = dist;
                end
            end
            NN = minDist;
        end
        
        function KNN = find_KNN(x, y, cluster)
            % calculates distance to mean of k-nearest neighbours
            %function to return mean of k nearest neighbours
            k = 5;
            dist = [];
            for i =  1:size(cluster, 1)
                dist = [dist; sqrt((x-cluster(i, 1))^2+(y-cluster(i, 2))^2) i];
            end
            minDist = sortrows(dist);
            minDistIndex = [];
            for i = 1:k
                minDistIndex = [minDistIndex minDist(i, 2)];
            end
            minXVals = [];
            minYVals = [];
            for i = 1:k
                minXVals = [minXVals cluster(minDistIndex(i), 1)];
                minYVals = [minYVals cluster(minDistIndex(i), 2)];
            end
            KNN = sqrt((x-mean(minXVals))^2+(y-mean(minYVals))^2);
        end

        function figureNumber = plotterFull(grid, dimension, classes, plotTitle, colourScheme, nuM)
            % standardized plotting
            colorDot = [[0 0 1]; [1 0 0]; [0 153/255 76/255]];
            figure(nuM);
            colormap(colourScheme);
            contourf(grid(:,:,1), grid(:,:,2), grid(:,:,dimension)); hold on;
            for i = 1:size(classes,1)
                class2Plot = classes(i);
                scatter(class2Plot.cluster(:,1), class2Plot.cluster(:,2), 20, colorDot(i,:),'filled'); hold on;
                plot(class2Plot.stdContour(:,1), class2Plot.stdContour(:,2), 'LineWidth', 2,'color',colorDot(i,:)); hold on;
            end
            xlabel('Feature 1');
            ylabel('Feature 2');
            title(plotTitle);
            figureNumber = nuM + 1;
        end
    end
end

