clc;

%Weights = ones(1,D);        % let Weights be the weights of the attributes
I = 2;                      % let I be the index of the first song
C_w = 1;                    % let C_w be the weight constant for multiplication
K_w = 10;                   % let K_w be the weight reduction factor. this is used to make sure that the magnitude of the weights remains checked

WeightMin = 0.01;           % let WeightMin be the minimum weight that can be assigned
% similarity S is defined as cos(theta) - tan_inverse|distance|/(pi/2)
% this takes into account the direction of the vector as well as the
% magnitude (second term was added for that)

Similarity = zeros(N,N);    % let Similarity be the similarity measure matrix for the edges connecting songs   

Feedback = -1;              % let Feedback be a boolean representation of positive or negative user Feedback. in continuous form, -1 <= Feedback <= 1

Suggestions = 10;           % let Suggestions be the number of songs we will suggest (number of iterations of the algorithm
                            % let SuggesitonsList be the list of indices of suggested songs
SuggestionsList = zeros(1,Suggestions);            

TimesPlayed = ones(1,N);    % the number of times each song has been played (offset by 1 i.e. not set to 0 for divisibility)

% pheromone deposition is currently independent of value associated with
% the edge. this is for better control of the system.

Pheromone = zeros(1,N);     % pheromone deposits on songs which have been played
PhMax = 1;                  % maximum value of pheromone
Lambda = 0.1;               % let Lambda be the pheromone evaporation factor

% let Probability be the probability matrix. each element in the ith row
                            % represents the probability of jumping to that song from the song corresponding to that row
                            % initial distribution is uniform
Probability = ones(N,N)/(N-1);   
K = floor(sqrt(2*N));       % let K be the number of candidate songs for the next song (dyncamic branching factor). K <= N-1. can be taken as floor(sqrt(2N)) in a manner similar to "rule of thumb" for number of froups
Candidates = zeros(1,K);    % let Candidates be the vector of indices of candidate songs to be chosen as next song
Alpha = 1;                  % let Alpha be the improvement rate
Gamma = 1;                  % let Gamma be the degradation rate
Delta = 0.05;               % let Delta be the incrimental change in probability

%%

% testing measures

precision = 0;              % let precision be the fraction of songs suggested which are in context of the current song being played

%%

%
% initialize the probabilities and playabilities as skew
%

for i = 1:N
                            % a song can not redirect to itself
    Probability(i,i) = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% play the selected song (corresponding to index I) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TimesPlayed(I) = TimesPlayed(I) + 1;

% deposit pheromone on Candidates(I)
Pheromone(I) = PhMax;

%%

%%%%%%%%%%%%%%%%%%%
% suggest song(s) %
%%%%%%%%%%%%%%%%%%%

suggestion_count = 1;

while(suggestion_count < (Suggestions+1))

    %
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % this is the beginning of a selection iteration %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %

    %%%%%%%%%%%%%%%%%%%%%%%%
    % select the next song %
    %%%%%%%%%%%%%%%%%%%%%%%%

    %
    % select K songs to select the next song from %
    %

    i = 0;

    Selectability = ones(1,N);  % so that no candidate is repeated in the current candidate list
    Selectability(I) = 0;
   
    % initialize variables
    
    Candidates = zeros(1,K);
    
    % modify probabilities based on pheromone values
    % maintain consistency so that the sum is 1

    pheromone_contribution = (PhMax - Pheromone)./TimesPlayed;

    pheromonized_probability = Probability(I,:).*pheromone_contribution;
    pheromonized_probability = pheromonized_probability /sum(pheromonized_probability);
    
    % select K candidates
    
    while(i < K)
        % select a song at random

        r = rand(1,1);          % generate a random number

        % get the index corresponding to this number

        index = 0;
        p_cumulative = 0;

        %iterate through the songs and select one
        
        for j = 1:N             % TODO: should we use cumulative probability instead and use binary search here?
                                % calculate new accumulative probability
            p_cumulative = p_cumulative + pheromonized_probability(j);
            
            if(p_cumulative > r)
                if(Selectability(j) > 0)
                    index = j;  % got the index

                    i = i + 1;  % found a candidate. now move on
                                % add it to the list of candidates
                    Candidates(i) = index;  
                                % to avoid repetition of songs in the suggestion list
                    Selectability(j) = 0;
                    break;
                end
            end
        end
    end

    %
    % evaluate candidates
    %

                            % vector for stroring similarity values of candidates
    candidate_similarity = zeros(1,K);

    for i = 1:K
                            % vectors taken for element I and Candidates(i)
        vector_1 = Attributes(I,:);
        vector_2 = Attributes(Candidates(i),:);

        cos_theta = dot(vector_1, vector_2)/(norm(vector_1)*norm(vector_2));

        % distance is weighted euclidian distance

                            % vector_diff_sq will only contain positive values
        vector_diff_sq = (vector_1 - vector_2).^2;
        distance = sqrt(sum(vector_diff_sq.*Weights));

        % now calculate similarity
        candidate_similarity(i) = cos_theta - (atan(distance)/(pi/2));
    end
    
    % get the index corresponding to the maximum similarity and suggest that
    % song
                                % index is relative only to the vector of similarity vector (not actual values of candidates)
    best_similarity_index = find(candidate_similarity == max(candidate_similarity));  
                                % in case it is an array, we take the first element
    best_similarity_index = best_similarity_index(1);
    group_I = find(result.data.f(I,:) == max(result.data.f(I,:)));

    Candidates(best_similarity_index)
    result.data.f(Candidates(best_similarity_index),group_I)
    
    
    
    
    
    
    
    
    
    
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % play the selected song (corresponding to index Candidates(best_similarity_index)) %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    TimesPlayed(Candidates(best_similarity_index)) = TimesPlayed(Candidates(best_similarity_index)) + 1;
    
    % deposit pheromone on Candidates(best_similarity_index)
    Pheromone(best_similarity_index) = PhMax;
    
    %%
    
    % take feedback
    if(result.data.f(Candidates(best_similarity_index),group_I) > 0.5)
        Feedback = 1;
        precision = precision + 1;
        
        % move to that song
        I = best_similarity_index;
    else
        Feedback = -1;
    end
    
    %%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % modify probability based on Feedback %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                                    % probabilities specific to candidates
    probability_subset = Probability(I,Candidates);
    
    if(Feedback ~= 0)
                                    % probabilities specific to candidates
        probability_subset = zeros(1,K);
                                    % get the subset
        for i = 1:K
            probability_subset(i) = Probability(I,Candidates(i));
        end

        %
        % calculate probability formula parameters a,b,c,d
        %

        pow_a = (1+Feedback)/2;
        pow_b = (1-Feedback)/2;
        pow_c = (1-sign(Feedback))/2;
        pow_d = (1+sign(Feedback))/2;

        c_1 = 10;
        c_2 = 10;
        
        %
        % calculate new probabilities
        %

        probability_sum = sum(probability_subset);

        modified_probability_subset = zeros(1,K);
     
        for i = 1:K             % this can later be made faster by first assigning for all elements and then assigning for the selected variable again
            if(i == Candidates(best_similarity_index))
                modified_probability_subset(i) = probability_subset(i) + c_1*(((Alpha*Delta)^pow_a)*((Gamma*Delta)^pow_b)*((-1)^pow_c));
            else
                modified_probability_subset(i) = probability_subset(i) + c_2*(((Alpha*Delta)^pow_b)*((Gamma*Delta)^pow_a)*((-1)^pow_d));
                                % make sure we have not subtracted too much
                if(modified_probability_subset(i) <= 0)
                                % assign it to a reset value
                    modified_probability_subset(i) = Delta;
                end
            end
        end


        % scale the probabilities to maintain integrity in original probability
        % vector

        modified_probability_subset_sum = sum(modified_probability_subset);

        new_probability_subset = modified_probability_subset*(probability_sum/modified_probability_subset_sum);

        %
        % assign the values back to the main vector
        %

        for i = 1:K
            Probability(I,Candidates(i)) = new_probability_subset(i);
        end       
    end
      
    %%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % modify weights based on Feedback %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    attribute_distances = Attributes;

    for i = 1:N
        attribute_distances(i,:) = abs(attribute_distances(i,:) - attribute_distances(I,:));
    end

    attribute_distances_avg = sum(attribute_distances)*(1/N);

    attribute_distances_selected = abs(attribute_distances(Candidates(best_similarity_index),:) - attribute_distances(I,:));

    % now recaliberate the weights

    % if the selected song got positive Feedback
    % if d_s < d_i (i.e. correct is closer) => w increases
    % w = w + C_w*(d_a - d_s)

    % if the selected song got negative Feedback
    % if d_s < d_i (i.e. incorrect is closer) => w decreases
    % w = w - C_w*(d_a - d_s)

    % Weights = Weights + C_w*Feedback*(attribute_distances_avg - attribute_distances_selected);

    if(Feedback == 1)
        Weights = Weights + C_w*(attribute_distances_avg - attribute_distances_selected);
                                    % make all values positive
        Weights = Weights - min(Weights) + WeightMin;
    else
        Weights = Weights - C_w*(attribute_distances_avg - attribute_distances_selected);
                                    % make all values positive
        Weights = Weights - min(Weights) + WeightMin;
    end

    % reduce the weights to smaller values
    Weights = Weights/K_w;

    %%
    
    %%%%%%%%%%%%%%%%%%%%
    % modify Pheromone %
    %%%%%%%%%%%%%%%%%%%%

    %
    % evaporate pheromone from all edges
    %
    
    Pheromone = Pheromone*exp(-Lambda);

    %%
    
    % one suggestion iteration completed    
    SuggestionsList(suggestion_count) = best_similarity_index;
    suggestion_count = suggestion_count + 1;
end

precision = precision/Suggestions;
precision
