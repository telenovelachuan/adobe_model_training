A data science project on fitting the training dataset and make predictions on the test set.

## Feature exploring

Take a look at the feature values and their distributions

- feature value scatter with y:
![feature_with_y](https://github.com/telenovelachuan/adobe_model_training/blob/master/reports/figures/features_with_y.png)

- boxplots for features and y
![boxplots](https://github.com/telenovelachuan/adobe_model_training/blob/master/reports/figures/boxplots.png)

- feature value distributions
![feature_dist](https://github.com/telenovelachuan/adobe_model_training/blob/master/reports/figures/feature_distr.png)

- look at the pearson correlations between features
![pearson_corr_orig](https://github.com/telenovelachuan/adobe_model_training/blob/master/reports/figures/pearson_corr_original.png)

From the pearson correlations, it could be observed that the most correlated features are: e, c_blue, and h_black. So I tried a few new features for better modeling as follows:
	- e * h * c
	- e * (h + c)
	- f + g
	- (f + g) * h * c
	- (f + g) * (h + c)
	- e * h * c * (f + g)
	- e * (h + c) * (f + g)
	- e * h * c + f + g
	- e * (h + c) + f + g
	- e * h * (f + g) * (h + c)
	- e * h + (f + g) * (h + c)
	- e * (h + c) * (f + g) * (h + c)
	- e * (h + c) + (f + g) * (h + c)

The pearson correlations for all the new features:
![pearson_corr_new](https://github.com/telenovelachuan/adobe_model_training/blob/master/reports/figures/pearson_corr_new.png)

[Click me for more details of feature exploration](https://github.com/telenovelachuan/adobe_model_training/blob/master/notebooks/feature%20exploration.ipynb)

## Regression modeling

I tried a stacking method that first fine tuned 5 base regression models on the training set, then a meta model is applied on these base models to fit the residual between the predictions by these base models and target values.

The below base model options are trained and tested using grid search and mean absolute percentage error as metric.

	- Random Forest
	- Kernel Ridge
	- Lasso regression
	- Elastic Net
	- Bayesian Ridge
	- XGBoost
	- GradientBoosting Regression
	
For these models, the 5 base models with best performance by 10-fold cross validation are Kernel Ridge, Lasso, Elastic Net and Bayesian Ridge.

The meta model is used to further fit the prediction errors by these fine tuned base models. Lasso outperformed all other options here as the meta model. 

I also tried a simple average of all the base models as output, which seemed to outperform the stacking method a little bit.

[Click me for more details of regression modeling](https://github.com/telenovelachuan/adobe_model_training/blob/master/notebooks/regression.ipynb)


 