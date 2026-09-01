from pathlib import Path
import json, joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data/raw/housing.csv"; RESULTS=ROOT/"results"; ASSETS=ROOT/"assets"; MODELS=ROOT/"models"; SEED=42
def split():
 d=pd.read_csv(RAW); y=d.pop("median_house_value"); return (*train_test_split(d,y,test_size=.2,random_state=SEED),)
def prep(x):
 n=x.select_dtypes(include="number").columns; c=x.select_dtypes(exclude="number").columns
 return ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),n),("cat",OneHotEncoder(handle_unknown="ignore"),c)])
def scores(name,m,x,y):
 p=m.predict(x); return {"model":name,"mae":mean_absolute_error(y,p),"rmse":mean_squared_error(y,p)**.5,"r2":r2_score(y,p),"mape_pct":np.mean(np.abs((y-p)/y))*100}
if not RAW.exists(): raise FileNotFoundError("housing.csv not found")
for p in (RESULTS,ASSETS,MODELS): p.mkdir(exist_ok=True)
x_train,x_test,y_train,y_test=split()
models={"Dummy mean":DummyRegressor(),"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(n_estimators=250,min_samples_leaf=2,n_jobs=-1,random_state=SEED),"Gradient Boosting":GradientBoostingRegressor(random_state=SEED)}
rows=[]; fitted={}
for name,est in models.items():
 m=Pipeline([("prep",prep(x_train)),("model",est)]).fit(x_train,y_train); fitted[name]=m; rows.append(scores(name,m,x_test,y_test))
table=pd.DataFrame(rows).sort_values("mae"); final=fitted[table.iloc[0].model]; joblib.dump(final,MODELS/"house_price_model.joblib"); table.to_csv(RESULTS/"model-comparison.csv",index=False); (RESULTS/"metrics.json").write_text(json.dumps(table.iloc[0].to_dict(),indent=2),encoding="utf-8")
plt.hist(y_train,bins=40); plt.title("Target distribution"); plt.tight_layout(); plt.savefig(ASSETS/"target-distribution.png",dpi=160); plt.close()
p=final.predict(x_test); residual=y_test-p; plt.scatter(p,residual,alpha=.3); plt.axhline(0,color="red"); plt.xlabel("Predicted price"); plt.ylabel("Residual"); plt.tight_layout(); plt.savefig(ASSETS/"residual-plot.png",dpi=160); plt.close()
plt.bar(table.model,table.mae); plt.xticks(rotation=20); plt.ylabel("MAE"); plt.tight_layout(); plt.savefig(ASSETS/"model-comparison.png",dpi=160); plt.close()
print(table.to_string(index=False))
