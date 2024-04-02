<div align="center">
<h1>
  MiniMATH Report
</h1>
</div>

<h4 align="center">
    <p>
        <a href="https://github.com/jzx-ai-lab/MiniMATH/blob/main/report.md">中文</a> | <b>English</b>
    <p>
</h4>

### Preface
The fervent exploration of large-scale models has been ongoing for over a year, catalyzed by the public unveiling of OpenAI's GPT-3.5. A pivotal concern within the research community has been the extent of knowledge that models of certain parameter sizes (e.g., 7B, 13B, 72B) can encapsulate and the breadth of their general linguistic capabilities. Given the inefficiencies inherent in the recursive computational methodologies of Seq2Seq task, the correlation between model parameters and capabilities directly influences the cost-effectiveness of deploying these models in real-world production scenarios. It is evident that the compression of model parameters and capabilities represents a core competitive advantage in the emergence of a definitive leader in the large model domain.

Since OpenAI initiated discussions on the phenomenon of emergence, propelling the standard research dimensions of large models to 7B+ parameters, there has been a conspicuous absence of further public disclosures regarding the technological advancements or iterations of GPT-3.5-Turbo or GPT-4-preview. Consequently, the industry's understanding of model compression outcomes largely stems from speculative analysis. However, Microsoft's introduction of the 'tiny' and 'phi' series models, alongside the publication of the "Textbooks Are All You Need" paper, has reinvigorated the data-centric approach to Large Language Models (LLMs), inspiring subsequent research into compact models such as Gemini-Nano, Qwen1.5, and MiniCPM, which have demonstrated remarkable performance.

The Phi-1 model, trained on 7B tokens for 7-8 epochs and sized at 1.3B, outperformed the majority of models within the 7-13B range on the HumanEval benchmark, equating its performance with that of GPT-3.5. This achievement preliminarily validated the hypothesis that high-quality data, even when reduced in volume and computational demand, can surpass the State of the Art (SOTA) established by larger models in specific tasks. Subsequent models, Phi-1.5 and Phi-2, built upon the successful foundation of Phi-1, leveraging extensive synthetic data for training, thereby achieving superior performance across a range of general tasks.

The strategic use of synthetic data, as highlighted in the "Textbooks Are All You Need" series, emerges as a critical element. An analysis of substantial real-world code data revealed numerous impediments to model learning, such as missing code modules, prevalence of non-functional code, and the absence of explanatory documentation for complex algorithms. This analysis underscored the belief that a concise, self-contained, instructive, and evenly distributed dataset could significantly enhance model learning. Given the challenges associated with extracting high-quality data from real-world sources and the potential biases introduced during data curation, synthetic data emerges as a pivotal solution.

The singular focus and straightforward synthetic data generation process of Phi-1, contrasted with the unelaborated methods of constructing extensive data for general tasks in Phi-1.5 and Phi-2, lend credence to the notion that synthetic data generation will play a crucial role in the future trajectory of LLMs. Unlike the natural sciences, where data is guided by rich theoretical descriptions or continuous real-world observations, language represents a form of discrete data, a higher-dimensional descriptor of various modalities created by humans rather than derived from nature. The absence of strict rules governing language, coupled with the diverse perceptions across different cultures, ethnicities, and individuals, presents a complex data landscape for language-based tasks. Determining the nature of effective synthetic data, the methodologies for its generation, and the criteria for evaluating its efficacy remains a profound area of inquiry, poised to be a decisive factor in the latter stages of LLM development.

Recognizing the potential of synthetic data and small-parameter models for practical applications, preliminary training endeavors have been conducted within the educational sector, utilizing existing high-quality data. Through iterative adjustments in data synthesis and proportioning, the MiniMATH-1.5B-base model was developed, laying the groundwork for future enhancements in LLM capabilities and data quality.

## Introduction

The MiniMATH-1.5B-base model is a pre-trained model that has been developed based on a curated dataset comprising 19 billion mathematical problems, textual content, and dialogues. This model has achieved commendable results in logical reasoning on Chinese mathematical problems without undergoing processes such as Sparse Fine-Tuning (SFT) or Data Poisoning Optimization (DPO), and under conditions where test data were strictly segregated. Additionally, the model exhibits capabilities akin to aligned models in dialogue and following instructions, despite the absence of techniques such as learning rate scheduling in its training regimen. Instead, the training process involved only two-stage batch size scheduling and four-stage data ratio adjustments. Through performance experiments conducted at different stages, the significance of high-quality and well-balanced data in enhancing the generative capabilities of the model was affirmed.

### Limitations:
- Due to the constraints imposed by the model's scale and the simplistic pretraining methodology employed, the model may exhibit issues related to hallucinations and echolalia.
- The model has been predominantly trained on high-quality Chinese content due to limitations in data scale, which may result in suboptimal performance when handling English queries.
- The performance of the model in multi-turn dialogues may be adversely affected by the reliance on pretraining methodologies.
- The pretraining approach may predispose the model's responses to verbosity. Furthermore, the constraints related to the model's scale may exacerbate issues associated with hallucinatory responses.

## Performance
|model|Non-Emb Params|C-Eval|CMMLU|MMLU|GSM8K|MATH|MATH23K|Math Avg.|training-token|
|-|-|-|-|-|-|-|-|-|-|
|TinyLlama-1.1B|1.1B|25.02|24.03|24.3|2.27|0.74|-|1.5|~1T
|Qwen1.5-1.8B|1.2B|59.7|57.8|46.8|38.4|10.1|58.3|35.6|-|
|Gemini Nano-3B|-|-|-|-|22.8(report)|-|-|-|-|
|StableLM-Zephyr-3B|2.7B|30.34|30.89|45.9|52.54|12.49|37.68|32.5|-|
|Phi-1.5|-|-|-|42.87|40.2|-|-|-|150B 
|Phi-2-2B|2.5B|23.37|24.18|52.66|57.16|3.5|43.39|30.35|1.4T
|MiniCPM-2B|2.4B|51.13|51.07|53.46|53.83|10.24|67.5|43.83|~1T
|MiniMATH-1.5B|1.2B|41.64|42.70|29.13|43.34|14.5|60.04|39.29|58B

***The token count refers to the total number of tokens used for training.**

***During the MiniMATH training process, the epochs are divided into four distinct stages.**

## Vocabulary
Vocab used bt MiniMATH belongs to qwen-72B，The vocabulary for this series is expanded based on OpenAI's cl100k_base, resulting in a vocabulary size of 150k. The OpenAI vocabulary utilizes the titoken tokenization algorithm, which offers faster tokenization speeds compared to the original BPE.

## Training details
During the training process of MiniMATH, the learning rate was maintained at a constant 2e-4, and the dataset was segmented into four distinct stages. The proportion of data transitioned gradually from general to specialized texts across these stages, aiming to infuse specialized knowledge into the model while preserving its general capabilities. Training with general texts facilitates the model's rapid convergence on specialized tasks. Furthermore, given that the parameter volume of the embedding input and output layers constitutes a significant proportion in smaller models, to enhance the effective parameter volume of the model, the weights of the input and output layers were shared. This approach aligns with advanced practices in the field.

### 1. Initialize
In accordance with the recommendations delineated in the GPT-2 manuscript, the initialization of the linear projection layer associated with residual connections was undertaken utilizing a normal distribution with a mean of zero and a standard deviation of $\frac{0.02}{\sqrt{2 \times \text{layers}}}$, where the parameter 'layers' corresponds to the total number of transformer layers within the model. For the initialization of the remaining layers, a normal distribution characterized by a mean of zero and a standard deviation of 0.02 was employed.

### 2. Datasets
The training regimen of MiniMATH utilized a deduplicated token collection amounting to 19.24 billion. The data was segmented into four distinct stages, each characterized by its unique ratio, aiming to optimize performance across various downstream tasks related to mathematics.

### 3. Initial Stage
- #### 3.1 Data Composition
    In the initial stage, textual data predominantly sourced from several high-quality open-source datasets were employed, totaling 9.2 billion tokens. This stage underwent training for one epoch. The English language data primarily derived from the tiny_strange_text_book, a product of the phi series works, while the Chinese language content was chiefly sourced from the Chinese News section of wanjuan-1.0.
    
    ![](https://github.com/jzx-ai-lab/MiniMATH/blob/main/assets/img/stage-1-data-en.png)
- #### 3.2 Training Parameters and Loss Variation
    During the foundational training stage, a fixed learning rate of 2e-4 and a batch size of 163,840 were adopted. The rationale behind selecting a relatively smaller batch size for this stage stems from its conceptualization as a form of weight initialization process. A smaller batch size is thought to introduce a greater degree of variability to the initial weights, thereby better preparing the model for subsequent tasks.

    The loss variation curve for the initial stage is illustrated below. Notably, the loss rapidly converges to approximately 2.4 around the 10k step mark. This rapid convergence is attributed to the relatively modest total token count and the somewhat homogeneous distribution of the data, which also underpins the decision to limit this stage to a single epoch of training.
    ![](https://github.com/jzx-ai-lab/MiniMATH/blob/main/assets/img/stage-1-loss.png)

### 4. Stage II

- #### 4.1 Data Composition

    During the second stage, the entirety of the data from the first stage was retained. Subsequently, a cleaning process was undertaken on selected high-quality mathematical texts from sources such as Mathpile and Wikipedia, amounting to 0.899 billion. This resulted in an accumulated dataset of 10.12 billion, upon which training was conducted for one epoch.
    
    ![](https://github.com/jzx-ai-lab/MiniMATH/blob/main/assets/img/stage-2-data-en.png)

- #### 4.2 Training Parameters and Loss Variation

    The training parameters in the second stage remained unchanged from the first stage. The variation in loss is illustrated in the figure below. Upon transitioning to the second stage, there was no significant change observed in the loss. After undergoing training in the second stage, the loss gradually decreased to the vicinity of 2.0.
    ![](https://github.com/jzx-ai-lab/MiniMATH/blob/main/assets/img/stage-2-loss.png)

### 5. Stage III

- #### 5.1 Data Composition

    In the third stage, while retaining a certain amount of data from stages 1 and 2, a substantial volume of math-related question data, amounting to 3.45 billion (totaling 4.82 billion), was introduced. This included publicly available datasets such as Math23k and Ape210k, along with proprietary high-quality question data. For the majority of the open-source data, a Chain of Thought (COT) problem-solving process was constructed, thereby enhancing the small model's capability for extended logical reasoning. This stage involved training for one epoch.
    
    ![](https://github.com/jzx-ai-lab/MiniMATH/blob/main/assets/img/stage-3-data-en.png)

- #### 5.2 Training Parameters and Loss Variation

    In this stage, aiming to achieve a lower loss on question data, and drawing inspiration from the Phi series and MiniCPM series work, the batch size was increased tenfold to 1.63 million. The rest of the training parameters remained consistent. A comparison was made between the loss before and after the batch size increase, as well as the accuracy on actual downstream tasks.

    During the transition from stage 2 to stage 3, the loss experienced a sudden decrease from the vicinity of 2.0 to around 1.5. After two epochs of training, it eventually converged to around 1.2. This indicates that, following the weight adjustments in the initial two stages, the model rapidly adapted to the downstream tasks. The adjustment in batch size during this stage did not exhibit a significant difference in loss variation.

    <h4 align="center">
        bathsize = 163K
    </h4>

    ![](https://github.com/jzx-ai-lab/MiniMATH/blob/main/assets/img/stage-3-loss-1.png)

    <h4 align="center">
        bathsize = 1.63M
    </h4>

    ![](https://github.com/jzx-ai-lab/MiniMATH/blob/main/assets/img/stage-3-loss-2.png)

    In order to assess the impact on performance in specific downstream tasks, an evaluation was conducted on the math23k-test dataset to observe the variations in problem-solving accuracy as a function of different batch sizes. It was ultimately observed that modifications in the batch size resulted in an approximate increase of 1.3 percentage points in accuracy. Consequently, it has been decided to continue employing a batch size of 1.63M in the subsequent stage of training.

    |model|eval-dataset|score|
    |-|-|-|
    |MiniMATH_BS_163K-epoch-2|matk23k-test-1k|25.6|
    |MiniMATH_BS_1.63M-epoch-2|matk23k-test-1k|26.9|

### 6. Stage IV

- #### 6.1 Data Composition

    In the fourth stage, a partial selection of data from stages 1 and 2 was retained alongside the entirety of the data from stage 3. Additionally, a substantial volume of data pertaining to basic mathematical operations was incorporated. This included simple arithmetic operations (addition, subtraction, multiplication, and division), elementary linear equation solving processes, and basic mathematical question-and-answer data derived from the MNBVC project, as well as mathematical COT data amounting to 6.48B, culminating in a total of 11.3B. This stage underwent training over the course of two epochs.
    
    ![](https://github.com/jzx-ai-lab/MiniMATH/blob/main/assets/img/stage-4-data-en.png)

- #### 6.2 Training Parameters and Loss Variation

    The training parameters in the fourth stage remained consistent with those of the third stage, with a batch size of 1.63M and a learning rate of 2e-4. Throughout this stage, the variation in loss was relatively gradual, decreasing slowly from an initial value of approximately 1.25 to about 1.13.
    ![](https://github.com/jzx-ai-lab/MiniMATH/blob/main/assets/img/stage-4-loss.png)

    The inclusion of basic mathematical data played a pivotal role in enhancing the accuracy of problem-solving. After incorporating basic mathematical data and conducting training for two epochs, MiniMATH achieved a problem-solving accuracy rate of 60.04% on the matk23k-test-1k dataset, marking an increase of 33.14 percentage points. Furthermore, it attained scores of 41.64 and 42.7, respectively, on the C-Eval and CMMLU datasets, which are utilized for evaluating general capabilities.

    |model|eval-dataset|score|
    |-|-|-|
    |MiniMATH_BS_1.63M-epoch-5|matk23k-test-1k|60.04|
    |MiniMATH_BS_1.63M-epoch-5|C-Eval|41.64|
    |MiniMATH_BS_1.63M-epoch-5|CMMLU|42.7|

## Demonstration of Model Capabilities
### 1. Mathematical Problem-Solving Ability
|Query|Response|
|-|-|
|On Teacher's Day, out of the 42 students in Class 6(1), (1/3) of the students arrived early to decorate the classroom, and the number of students who participated in making greeting cards for the teacher was (8/7) times the number of students who decorated the classroom. How many students participated in making greeting cards?|First, let's find out how many students came early to arrange the classroom. Since 1/3 of the students came early, we calculate:<br>(1/3) * 42 = 14 students<br>Next, let's find out how many students participated in making greeting cards. Since the number of students who participated in making greeting cards was 8/7 times the number of students who arranged the classroom, we calculate:<br>(8/7) * 14 = 16 students<br>Therefore, 16 students participated in making greeting cards.|
|A ship sails from Port A to Port B at a speed of 45 km/h and arrives 6 hours later. When the ship returns from Port B to Port A, it takes 1 hour less. What is the speed of the ship on its return journey?|Let's denote the distance from Port A to Port B as \( D \) km.<br>When the ship is traveling from Port A to Port B at 45 km/h, it takes 6 hours to reach port B. This means that the distance \( D \) can be calculated as:<br>\[ D = 45 \text{ km/h} \times 6 \text{ hours} \]<br>\[ D = 270 \text{ km} \]<br>When the ship is traveling from Port B to Port A, it takes 1 hour less. This means that the time it takes to reach port A is 5 hours. We can calculate the speed of the ship on its return journey by dividing the distance by the time:<br>\[ \text{Speed return} = \frac{\text{Distance}}{\text{Time return}} \]<br>\[ \text{Speed return} = \frac{270 \text{ km}}{5 \text{ hours}} \]<br>\[ \text{Speed return} = 54 \text{ km/h} \]<br>Therefore, the speed of the ship on its return journey is 54 km/h.|
|A steel mill plans to produce 80,000 tons of steel this year, which is increase 1/4 compare with last year. How much steel did it produce last year?|To find out how much steel was produced last year, we need to divide the total amount of steel produced this year by 1.25 (since it is a 1/4 increase from last year).<br>So, the amount of steel produced last year is:<br>80,000 tons / 1.25 = 64,000 tons<br>Therefore, the steel mill produced 64,000 tons of steel last year.|
### 2. Mathematical Knowledge Explanation Ability
|Query|Response|
|-|-|
|Can you explain arithmetic progression for me?|Sure! Arithmetic progression is a sequence of numbers where each term is obtained by adding a constant value to the previous term. If the constant value is constant, then the sequence is an arithmetic sequence. The first term of the sequence is the first term of the sequence, and the common difference between consecutive terms is the same value.<br>To calculate the arithmetic progression, you can use the formula:<br>\[a_n = a_1 + (n - 1)d\]<br>where \(a_n\) is the nth term of the sequence, \(a_1\) is the first term, and \(d\) is the constant difference.<br>For example, if the first term \(a_1 = 5\) and the constant difference \(d = 2\), then the 10th term \(a_{10}\) can be calculated as follows:<br>\[a_{10} = 5 + (10 - 1) \cdot 2\]<br>\[a_{10} = 5 + 9 \cdot 2\]<br>\[a_{10} = 5 + 18\]<br>\[a_{10} = 23\]<br>So, the 10th term of the sequence is 23.<br>What's the topic you want to discuss?|
|What characteristics does addition has？|Addition is a binary operation that involves combining two or more numbers by adding their corresponding digits. It has several general properties, including commutativity, associativity, distributivity, and inverses. It also has applications in mathematics, computing, and everyday life.|
|Can you explain linear equation for me ? |Of course! Here is a linear equation for a straight line: y = mx + b. The variables are the coordinates of the point on the line, and the constant m represents the slope of the line. The base of the line is the distance between the two points on the line, and the height of the line is the distance between the two points on the line. The slope m represents the rate at which the height changes with the change in the x-coordinate of the point.|

### 3. Math Problem-creating Ability
|Query|Response|
|-|-|
|Help me to create three math problems about statistics.|1. A company wants to survey its employees about their favorite color. They have 200 employees and each is asked to choose a color from 3 options: red, blue, or green. Based on the selection, the company will calculate the average number of employees who chose each color. How many employees chose each color, and what was the average number of employees who chose each color?<br>2. A company wants to analyze the relationship between the number of hours studied and the grades earned by its students. The company has 500 students and each is given 3 hours of study. Based on the study hours, the company will calculate the average grade earned by its students. How many students studied each hour, and what was the average grade earned by the students?<br>3. A company wants to know the average height of its employees. The company has 200 employees and each is given a height measurement. Based on the measurements, the company will calculate the average height of its employees. How many employees are measured, and what was the average height of the employees?
|Help me to create 10 random linear equations|Here are 10 random linear equations:<br>1. 2x + 3y = 8<br>2. 4x - 5y = 7<br>3. 3x + 2y = 10<br>4. 5x - 4y = 11<br>5. 7x + 2y = 15<br>6. 4x + 3y = 9<br>7. 2x - 3y = 7<br>8. 5x - 4y = 13<br>9. 3x + 4y = 14<br>10. 4x + 5y = 12|

### 4. General Problem Response Ability
|Query|Response|
|-|-|
|Can you tell me who was the first human astronaut to land on the moon?|The first human astronaut to land on the moon was Neil Armstrong.|
|How do you know about blue whale?|Blue whales are the largest mammals in the world, with a mean weight of 170 tons (80 kilograms) and a height of 10 meters (3.3 feet). They live near the ocean, are known for their unique behavior, and are known for their keen eyesight and sense of smell. Blue whales are also known for their ability to swim and are known for their incredible strength and strength of the legs. In addition, blue whales are known for their ability to communicate with each other through telepathic signals.|


## Future Expectations and Improvements
It is noteworthy that, under the premise of utilizing only a pre-training process and a training dataset comprising merely 58 billion tokens, our model has achieved commendable results in multidimensional capability assessments. It has surpassed all models of similar scale in mathematical abilities. Although there remains a gap in general capabilities when compared to the current state-of-the-art small-scale models in the industry, we are confident that with further effort on domain-specific data, our smaller model will exhibit substantial competitive strength in practical educational applications.

Based on the preliminary experimental outcomes and our future expectations, we intend to enhance our model's capabilities in the following areas:
- Continue the accumulation of high-quality, proprietary data, aiming for a total token count exceeding 100 billion, with no less than 30% of these tokens pertaining to educational content.
- Further refine the model structure, including but not limited to, reconstructing efficient Mixture of Experts (MoE) structures on smaller models, more effectively integrating verifier models within smaller models, and continually incorporating the industry's most advanced architectures into our model.