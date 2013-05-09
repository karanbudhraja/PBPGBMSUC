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

%%

% testing measures

precision = 0;              % let precision be the fraction of songs suggested which are in context of the current song being played

%%

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
    % initialize similarity matrix
    %

    for i = 1:N
        for j = i:N              
                                 % vectors taken for element I and Candidates(i)
            vector_1 = Attributes(i,:);
            vector_2 = Attributes(j,:);

            cos_theta = dot(vector_1, vector_2)/(norm(vector_1)*norm(vector_2));

            % distance is weighted euclidian distance

                                % vector_diff_sq will only contain positive values
            vector_diff_sq = (vector_1 - vector_2).^2;
            distance = sqrt(sum(vector_diff_sq.*Weights));
            
            % now calculate similarity
            Similarity(i,j) = cos_theta - (atan(distance)/(pi/2));
            
            % assign the value to the main matrix
            Similarity(j,i) = Similarity(i,j);
        end

        % normalize similarity value
        
        Similarity(i,i) = 0;
    end

    for i = 1:N
        Similarity(i,:) = Similarity(i,:)/sum(Similarity(i,:));
    end

    pheromone_contribution = (PhMax - Pheromone)./TimesPlayed;
    pheromonized_similarity = Similarity(I,:).*pheromone_contribution;
    
    % get the index corresponding to the maximum similarity and suggest that
    % song
                                % index is relative only to the vector of similarity vector (not actual values of candidates)
    %best_similarity_index = find(Similarity(I,:) == max(Similarity(I,:)));  
    best_similarity_index = find(pheromonized_similarity == max(pheromonized_similarity));  
                                % in case it is an array, we take the first element
    best_similarity_index = best_similarity_index(1);
    group_I = find(result.data.f(I,:) == max(result.data.f(I,:)));

    %%%%%%%%%%%%%%%%% DEBUG INFORMATION %%%%%%%%%%%%%%%%%%
    best_similarity_index
    result.data.f(best_similarity_index,group_I)
    
    v_1 = Attributes(I,:);
    v_2 = Attributes(best_similarity_index,:);

    cos_theta = dot(v_1, v_2)/(norm(v_1)*norm(v_2));
    v_d = (v_1 - v_2).^2;
    distance = sqrt(sum(v_d.*Weights));    
    %%%%%%%%%%%%%%%%% DEBUG INFORMATION %%%%%%%%%%%%%%%%%%
    
    %%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % play the selected song (corresponding to index Candidates(best_similarity_index)) %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    TimesPlayed(best_similarity_index) = TimesPlayed(best_similarity_index) + 1;
    
    % deposit pheromone on Candidates(best_similarity_index)
    Pheromone(best_similarity_index) = PhMax;
    
    %%
    
    % take feedback
    if(result.data.f(best_similarity_index,group_I) > 0.5)
        Feedback = 1;
        precision = precision + 1;
        
        % move to that song
        I = best_similarity_index;
    else
        Feedback = -1;
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

    attribute_distances_selected = abs(attribute_distances(best_similarity_index,:) - attribute_distances(I,:));

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
