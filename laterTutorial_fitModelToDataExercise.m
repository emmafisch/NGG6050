% LATER Fitting Exercise
%
% Copyright 2023 by Joshua I. Gold, University of Pennsylvania

% The basic idea in fitting a model to data is to find the parameters of
% the model that provide in some sense the best match of the model to the
% data. This match is provided by the "objective function." 
% This exercise is intended to demystify this process by getting you
% to define the initial conditions and objective function for 
% fitting the LATER model to RT data. For a much more thorough, but still 
% very accessible, overview of model fitting (to behavioral data), here
% is a great place to start:
%
% https://elifesciences.org/articles/49547
%
% For this exercise, recall that the point of the LATER model is that 1/RT is
% distributed as a Gaussian, where we can define the parameters
% of the Gaussian (mu and sigma) with respect to the standard parameters
% of the LATER model (muR and deltaS):
%       mu = muR/deltaS
%       sigma = 1/deltaS
%
% So fitting LATER to behavioral data involves finding parameters
% muR and deltaS that provide the best match to the data, according to 
% the appropriate objective function.
%
% Follow along the steps below, some of which will require you to complete
% the code (and therefore hopefully think about how to relate the high-
% level concepts discussed above with the nitty-gritty part of getting 
% everything to actually work.

%%  1. Get the data
%   
%   Use this code to get a data set (array of RTs from a single condition) 
%   to fit, already preprocessed to include correct trials only and remove
%   outliers (including express saccades). See later_getData for details
data = later_getData([], [], 0.2);
RTs = data{1};
clear data

global reciprocalRTs

%%  2. Define the objective function
%
% The objective function typically defines the error that you want to 
% minimize between your data and the model predictions. A common objective 
% function is the negative of the sum of the log-likelihoods of the data, 
% given the model parameters. To unpack that for the LATER model:
%
%   1. For each data point (RT from a single trial, in this case) and given
%       set of model parameters, compute the probability of the data, given
%       the model (i.e., the likelihood)
%   2. Take the logarithm
%   3. Sum all these log-likelihoods from all the data points
%   4. Take the negative, because we want to find the minimum (thus
%        corresponding to the maximum likelihood)
%
%   You can define the function simply using an "anonymous function"
%   (https://www.mathworks.com/help/matlab/matlab_prog/anonymous-functions.html), 
%   using this template that assumes that "fits" is a 2x1 vector of
%   [muR, deltaS]:
 
% EXERCISE:
% This is the objective function that defines the "fits" based on muR and deltaS
function nll = later_log_likelihood(fits, RTs)
    % 'fits' is the parameter vector [muR, deltaS] being optimized.
    muR = fits(1);    % Extract the first parameter: muR
    deltaS = fits(2); % Extract the second parameter: deltaS
    
    % Ensure deltaS is positive and avoids division by zero
    if deltaS <= 0
        nll = Inf;  % Return infinity if deltaS is non-positive (invalid)
        return;
    end

    % Compute the parameters for the LATER model based on muR and deltaS
    mu = muR / deltaS;      % Mean of the Gaussian
    sigma = 1 / deltaS;     % Standard deviation of the Gaussian
    
    % Invert the reaction times
    reciprocalRTs = 1 ./ RTs;
    
    % Compute the log-likelihood for each data point
    % Ensure that sigma is positive for valid normpdf calculation
    if sigma <= 0
        nll = Inf;  % Return infinity if sigma is non-positive (invalid)
        return;
    end
    
    % Calculate the negative log-likelihood
    log_likelihoods = log(normpdf(reciprocalRTs, mu, sigma));
    
    % Ensure that no NaN or Inf values are encountered
    if any(isnan(log_likelihoods)) || any(isinf(log_likelihoods))
        nll = Inf;  % Return infinity for invalid likelihoods
        return;
    end
    
    % Sum the log-likelihoods and negate it for minimization
    nll = -sum(log_likelihoods);
end



%%  3. Define initial conditions
%   
%   For the actual fitting, we will use fmincon
%   (https://www.mathworks.com/help/optim/ug/fmincon.html), which is 
%   "function minimization with constraints." This function allows for 
%   constraints that include upper and lower bounds on the parameters.
%   So here we define those bounds, along with the initial values.
%   We will use fairly arbitrary values for the lower and upper
%   bounds, but we should pick the initial values more judiciously. HINT: 
%   Recall that the muR and deltaS should be strongly related to 
%   empirical summary statistics of `the (reciprocal) RT distribution.
lowerBounds = [0.001 0.001];
upperBounds = [1000 1000]; 

% EXERCISE:
% Calculate empirical estimates for initial values
% Here, you could use mean and standard deviation of 1/RTs to set initial values.
muR_initial = mean(1 ./ RTs);  % Mean of 1/RTs
deltaS_initial = std(1 ./ RTs); % Standard deviation of 1/RTs

% Define initial values based on empirical data
initialValues = [muR_initial, deltaS_initial];

%%  4. Run the fits
% 
%   We will be using GlobalSearch . The general advantage of this approach 
%   is to avoid local minima; for details, see:
%   https://www.mathworks.com/help/gads/how-globalsearch-and-multistart-work.html
%  
%   These options seem to work well, but I dont have a stronger
%   rationale for using them. See the Matlab documentation if you really
%   want to dive in and understand them, and let me know if you find
%   better settings!
opts = optimoptions(@fmincon,    ... % "function minimization with constraints"
   'Algorithm',   'active-set',  ...
   'MaxIter',     3000,          ...
   'MaxFunEvals', 3000);

% Definine the "optimization problem" using variables defined above
reciprocalRTs = 1 ./ RTs;
err = -sum(log(normpdf(reciprocalRTs, fits(1)/fits(2), 1/fits(2))));
problem = createOptimProblem('fmincon',    ...
    'objective',   laterErrFcn,     ... % Use the objective function
    'x0',          initialValues,   ... % Initial conditions
    'lb',          lowerBounds,     ... % Parameter lower bounds
    'ub',          upperBounds,     ... % Parameter upper bounds
    'options',     opts);                % Options defined above

% Create a GlobalSearch object
gs = GlobalSearch;
   
% Run it, returning the best-fitting parameter values and the negative-
% log-likelihood returned by the objective function
[fits(ii,:), nllk] = run(gs,problem);

%%  5. Evaluate the fits
%
%   EXERCISE: How do you know if you got a reasonable answer?
% Display the results of the fitting process
fitted_muR = fits(1);
fitted_deltaS = fits(2);
disp(['Fitted muR: ', num2str(fitted_muR)]);
disp(['Fitted deltaS: ', num2str(fitted_deltaS)]);
disp(['Negative Log-Likelihood: ', num2str(nllk)]);

% Visualize the data and model fit (optional)
invRTs = 1 ./ RTs;
model_mu = fitted_muR / fitted_deltaS;
model_sigma = 1 / fitted_deltaS;

% Generate a histogram of 1/RTs (empirical data)
figure;
histogram(invRTs, 'Normalization', 'pdf');
hold on;

% Plot the Gaussian fit from the model
x_vals = linspace(min(invRTs), max(invRTs), 100);
gaussian_fit = normpdf(x_vals, model_mu, model_sigma);
plot(x_vals, gaussian_fit, 'r', 'LineWidth', 2);
title('Empirical Data and Model Fit (1/RT)');
xlabel('1/RT');
ylabel('Probability Density');
legend('Empirical Data', 'LATER Model Fit');
hold off;