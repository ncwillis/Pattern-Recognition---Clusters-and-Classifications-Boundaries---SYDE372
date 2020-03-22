classdef gridMaker   
    properties
        grid
        increment
    end
    
    methods
        function this = gridMaker(lxBound, rxBound, byBound, tyBound)
            this.increment = 500; % the 'pixel' density of the grid
            xVals = linspace(lxBound, rxBound, this.increment); % range of x values
            yVals = linspace(byBound, tyBound, this.increment); % range of y values
            [this.grid(:,:,1), this.grid(:,:,2)] = meshgrid(xVals, yVals); % generates a 3-D matrix, with the (:,:,1) representing row values and (:,:,2) representing column values
        end
    end
end

