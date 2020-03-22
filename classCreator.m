classdef classCreator
  
    properties
        caseNumber
        sampleSize
        mu
        sigma
        cholSigma
        cluster
        testCluster
        stdContour
        prior
    end
    
    methods
        function this = classCreator(caseNum, samplesize, meanVector, covarianceMatrix, caseSampleSize)
            % Creating the 'class' object
            this.caseNumber = caseNum;
            this.sampleSize = samplesize;
            this.mu = meanVector;
            this.sigma = covarianceMatrix;
            this.cholSigma = chol(this.sigma);
            this.cluster = this.applyTransform(this.mu, this.sampleSize, randn(this.sampleSize,2));
            this.testCluster = this.applyTransform(this.mu, this.sampleSize, randn(this.sampleSize,2));
            this.stdContour = this.cont();
            this.prior = this.sampleSize/caseSampleSize;
        end
        
        function sdContour = cont(this)
            % make a circle
            theta = 0:pi/5000:2*pi;
            x = cos(theta);
            y = sin(theta);
            unitCircle =[x(:), y(:)];
            % apply transform to circle
            sdContour = this.applyTransform(this.mu, length(unitCircle), unitCircle);
        end

        function transformedData = applyTransform(this, dataMean, dataSize, dataGen)
            transformedData = repmat(dataMean, dataSize, 1) + dataGen*this.cholSigma;
        end
    end
end

