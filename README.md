# Approximate Bayesian Computation MCMC
Many problems in data science involve estimating a set of parameters $\theta \in \Theta$ of a model $M$ that describes the processes underlying the problem of interest. In the Bayesian inference paradigm, the uncertainty over such a set of parameters is quantified by means of a posterior distribution, which is often described through sampling techniques. In this context, one assumes the parameters to follow a prior distribution $\pi(\theta)$, which outline the current believe
on the problem at hand, \textit{e.g.}, based on available expert knowledge. After having observed some data $\mathcal{D}$, this prior believe is updated by means of the likelihood function $\mathbb{P}(\mathcal{D}|\theta)$, which
describes the plausibility of having generated such data under all possible different values of $\theta$. The posterior distribution of interest, $f(\theta|\mathcal{D})$, is then determined by Bayes’ rule
\begin{equation*}
    f(\theta|\mathcal{D}) = \mathbb{P}(\mathcal{D}|\theta)\pi(\theta)/\mathbb{P}(\mathcal{D})
\end{equation*}
where $\mathbb{P}(\mathcal{D}) = \int_{\Theta}
\mathbb{P}(\mathcal{D}|\theta)\pi(\theta)d\theta$, called the evidence, represents the normalizing constant.
Stochastic simulation approaches for generating observations from the posterior distribution $f(\theta|\mathcal{D})$ often depend on knowing explicitly the likelihood function $\mathbb{P}(\mathcal{D}|\theta)$, possibly up to a multiplicative constant (\textit{i.e.} being able to evaluate it for any $\theta$ and $D$). However, for many complex probabilistic models, such likelihoods are either inaccessible or computationally prohibitive to evaluate, so one has to resort to the so-called likelihood-free methods, of which, most notably the \textbf{Approximate Bayesian Computation} (ABC). 

## Team:
Our team is composed by:   
- Di Gennaro Federico: [@FedericoDiGenanro](https://github.com/FedericoDiGennaro)  
- Lupini Michele:  
- Trentini Leonardo: [@leotrentini22](https://github.com/leotrentini22)  

## Environment:
You can find all the required packages and their version in the file `requirements.txt`.

## Description of notebooks
Here you can find what each file in the repo does. The order in which they are described follows the pipeline we used to obtain our results.
- `helpers.py`: implementation of all the "support" functions used in the main.
- `main.ipynb`: solution of all the points required by the project description.


