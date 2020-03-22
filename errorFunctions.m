classdef errorFunctions
    methods(Static)
	function closestIndex = findClosestIndex(x, y, gridX, gridY)
    	%send x grid, y grid: for grid(:,:,1), choose any row (get col index
    	%that is closest to x point). For grid(:,:,2) choose any col (get row
    	%index that is closes to y point)
    	xVals = gridX(1,:);
    	yVals = gridY(:,1);
    	[~, col] = min(abs(xVals-x));
    	[~, row] = min(abs(yVals-y));
    	closestIndex = [row col];
	end
	
	function error = getErrorCount(cluster, expectedVal, grid, dimension)
        %expected val is the label on the data
        grid = grid.grid; %Getting the grid from the grid object
    	errorCount = 0;
        % Counting incorrect classifications
    	for i = 1:size(cluster, 1)
        	index = errorFunctions.findClosestIndex(cluster(i,1), cluster(i,2), grid(:,:,1), grid(:,:,2));
        	xInd = index(1);
        	yInd = index(2);
        	if grid(xInd, yInd, dimension) == expectedVal
            		errorCount = errorCount +1;
        	end
    	end
    	error = errorCount;
	end
    end
end

