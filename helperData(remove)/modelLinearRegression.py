import numpy as np
import math as mt
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime, timedelta
import seaborn as sn
import itertools
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline   
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from ClassFunctions import SorteosTecLinealRegress, DNASColumn, BQLoad

mes=9
añosActual=2025

dfBoletosFisi=pd.read_csv("C:\Sam\Python\ScriptsMaquinaVirtual\Automatizacion\DataIntelligenceRepositoryData\Data_e2e\Transform\FCVentas_fisico.csv")
dfBoletosDig=pd.read_csv("C:\Sam\Python\ScriptsMaquinaVirtual\Automatizacion\DataIntelligenceRepositoryData\Data_e2e\Transform\FCVentas_digital.csv")
dfInfo=pd.read_csv("C:\Sam\Python\ScriptsMaquinaVirtual\Automatizacion\DataIntelligenceRepositoryData\Data_e2e\Transform\DMSorteos.csv")



dfBoletosDig=dfBoletosDig[["ID_SORTEO","ID_SORTEO_DIA","FECHAREGISTRO","CANTIDAD_BOLETOS","CANAL_DIG"]].rename({"CANAL_DIG":"CANAL"},axis=1)
dfBoletosFisi=dfBoletosFisi[["ID_SORTEO","ID_SORTEO_DIA","FECHAREGISTRO","CANTIDAD_BOLETOS","CANAL_TRADICIONAL"]].rename({"CANAL_TRADICIONAL":"CANAL"},axis=1)
dfBoletos=pd.concat([dfBoletosFisi,dfBoletosDig])

dfBoletos["FECHAREGISTRO"]=pd.to_datetime(dfBoletos.loc[:,"FECHAREGISTRO"],format='%Y-%m-%d')
dfInfo["FECHA_CIERRE"]=pd.to_datetime(dfInfo.loc[:,"FECHA_CIERRE"],format="%Y-%m-%d")



fecha_limite = datetime.now() - timedelta(days=3*365)
dfInfo = dfInfo[dfInfo["FECHA_CIERRE"] >= fecha_limite]

"""fecha_limite2 = datetime.now() - timedelta(days=1)
dfBoletos=dfBoletos[dfBoletos["FECHAREGISTRO"] <= fecha_limite2]"""


dfBoletos=pd.merge(dfBoletos,dfInfo[["ID_SORTEO","NOMBRE","FECHA_CIERRE","PRECIO_UNITARIO"]],on="ID_SORTEO",how="left")
dfBoletos["DNAS"]=DNASColumn(dfBoletos["FECHA_CIERRE"],dfBoletos["FECHAREGISTRO"])
dfBoletos= dfBoletos.sort_values("FECHAREGISTRO")
dfBoletos['CANTIDAD_BOLETOS_MEMBRESIAS'] = dfBoletos.apply(lambda row: row['CANTIDAD_BOLETOS'] if row['CANAL'] == 'Membresias' else 0, axis=1)
dfBoletos['CANTIDAD_BOLETOS_SIN_MEMBRE'] = dfBoletos.apply(lambda row: row['CANTIDAD_BOLETOS'] - row['CANTIDAD_BOLETOS_MEMBRESIAS'], axis=1)
dfBoletos=dfBoletos.drop("CANAL",axis=1)

dfBoletosDlasSin0=dfBoletos.groupby(["NOMBRE","DNAS"]).agg(CANTIDAD_BOLETOS=("CANTIDAD_BOLETOS","sum"),CANTIDAD_BOLETOS_SIN_MEMBRE=("CANTIDAD_BOLETOS_SIN_MEMBRE","sum"),CANTIDAD_BOLETOS_MEMBRESIAS=("CANTIDAD_BOLETOS_MEMBRESIAS","sum")).reset_index()
sorteos=dfBoletos["NOMBRE"].unique()
for sorteoVariable in sorteos:

    boletosDiasNegativos= dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] <= 0)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS_SIN_MEMBRE'].sum()
    dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] == 1)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS_SIN_MEMBRE'] += boletosDiasNegativos

    boletosDiasNegativos= dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] <= 0)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS_MEMBRESIAS'].sum()
    dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] == 1)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS_MEMBRESIAS'] += boletosDiasNegativos

    boletosDiasNegativos= dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] <= 0)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS'].sum()
    dfBoletosDlasSin0.loc[(dfBoletosDlasSin0['DNAS'] == 1)&(dfBoletosDlasSin0["NOMBRE"]==sorteoVariable), 'CANTIDAD_BOLETOS'] += boletosDiasNegativos

    # Eliminar las filas con días negativos
dfBoletosDlasSin0 = dfBoletosDlasSin0[dfBoletosDlasSin0['DNAS'] > 0]
dfBoletosDlasSin0=dfBoletosDlasSin0.sort_values("DNAS",ascending=False)
dfBoletosDlasSin0["BOLETOS_ACUMULADOS_SIN_MEMBRE"] = dfBoletosDlasSin0.groupby("NOMBRE")["CANTIDAD_BOLETOS_SIN_MEMBRE"].cumsum()
dfBoletosDlasSin0["BOLETOS_ACUMULADOS_CON_MEMBRE"] = dfBoletosDlasSin0.groupby("NOMBRE")["CANTIDAD_BOLETOS"].cumsum()


dfBoletosEscalado=pd.merge(dfBoletosDlasSin0,dfInfo[["SORTEO_GRUPO","NOMBRE","EMISION","ID_SORTEO"]],on="NOMBRE",how="left")
dfBoletosEscalado["PORCENTAJE_DE_AVANCE_SIN_MEMBRE"]=dfBoletosEscalado["BOLETOS_ACUMULADOS_SIN_MEMBRE"]/dfBoletosEscalado["EMISION"]
dfBoletosEscalado["PORCENTAJE_DE_AVANCE_CON_MEMBRE"]=dfBoletosEscalado["BOLETOS_ACUMULADOS_CON_MEMBRE"]/dfBoletosEscalado["EMISION"]


max_days = dfBoletosEscalado.groupby('NOMBRE')['DNAS'].transform('max')-1
dfBoletosEscalado["PORCENTAJE_DNAS"]=(max_days-(dfBoletosEscalado["DNAS"]-1))/max_days
dfBoletosEscalado=dfBoletosEscalado.sort_values(["NOMBRE","DNAS"],ascending=False)


class SorteosTecLRWM:
    sorteosSeleccionados = [] 
    def __init__(self,tipoProducto, mode,nombreSorteo = None, emision=None, fechaCelebra=None, sorteosEntrenamiento=None, X_=None, y_=None, data=None, dfInfoSorteos=None):
        self.mode = mode
        self.X_ = X_
        self.y_ = y_
        self.data = data
        self.dfInfoSorteos = dfInfoSorteos
        self.tipoProducto=tipoProducto

        if mode == "automatic":
            self.nombreSorteo= None
            self.emision = None
            self.fechaCelebra = None
            self.sorteosEntrenamiento = None
            self._set_sorteo_info()  # Asigna valores automáticamente

        elif mode == "manual":
            # Usa los valores proporcionados por el usuario
            self.nombreSorteo=nombreSorteo
            self.emision = emision
            self.fechaCelebra = fechaCelebra
            self.sorteosEntrenamiento = sorteosEntrenamiento

        else:
            raise ValueError("El modo debe ser 'automatic' o 'manual'.")
    
    @staticmethod
    def reset_sorteos():
        """Reinicia la lista de sorteos seleccionados."""
        SorteosTecLRWM.sorteosSeleccionados = []

    
    def _set_sorteo_info(self):
        if self.dfInfoSorteos is not None:
            # Convertir fechas a formato datetime
            self.dfInfoSorteos["FECHA_CIERRE"] = pd.to_datetime(self.dfInfoSorteos["FECHA_CIERRE"], errors="coerce")


            # Filtrar sorteos del tipo especificado
            dfSorteosEntrenamiento = self.dfInfoSorteos.loc[self.dfInfoSorteos["SORTEO_GRUPO"] == self.tipoProducto].copy()
            if dfSorteosEntrenamiento.empty:
                raise ValueError(f"No se encontraron sorteos para el grupo {self.tipoProducto}")

            # Normalizar las fechas
            dfSorteosEntrenamiento.loc[:, "FECHA_NORMALIZADA"] = dfSorteosEntrenamiento["FECHA_CIERRE"].apply(
            lambda x: x.replace(year=2000) if not pd.isnull(x) else None)

            # Filtrar sorteo actual
            


            dfSorteoActual = dfSorteosEntrenamiento.loc[dfSorteosEntrenamiento["FECHA_CIERRE"] >= dt.datetime.now()]
            dfSorteoActual = dfSorteoActual[~dfSorteoActual["NOMBRE"].isin(SorteosTecLRWM.sorteosSeleccionados)]

            if dfSorteoActual.empty:
                raise ValueError("No se encontró un sorteo actual en las fechas especificadas.")
            if len(dfSorteoActual)>1:
                print (f"Hay {len(dfSorteoActual)} sorteos activos de este grupo")
                dfSorteoActual=dfSorteoActual.sort_values("NUMERO_EDICION").head(1)

            

                

            # Calcular diferencias de días
            fechaSorteoActual = dfSorteoActual["FECHA_CIERRE"].max()
            if pd.isnull(fechaSorteoActual):
                raise ValueError("La fecha del sorteo actual es inválida.")
            
            dfSorteosEntrenamiento["DIAS_DIFF"] = abs(
                (fechaSorteoActual.replace(year=2000) - dfSorteosEntrenamiento["FECHA_NORMALIZADA"]).dt.days
            )
            dfSorteosEntrenamiento = dfSorteosEntrenamiento.sort_values("DIAS_DIFF") # Se ordenan de acuerdo a la cercania en fechas del sorteo actual
            dfSorteosEntrenamiento = dfSorteosEntrenamiento.iloc[0:4].sort_values("NUMERO_EDICION",ascending=False) #Se vuelven a ordenar para que vayan en orden de edicion

            # Asignar nombre del sorteo y sorteos de entrenamiento
            self.fechaCelebra=fechaSorteoActual
            self.emision=dfSorteoActual["EMISION"].item()
            self.nombreSorteo = dfSorteoActual["NOMBRE"].iloc[0]
            self.sorteosEntrenamiento = list(dfSorteosEntrenamiento.iloc[:, 3])

            SorteosTecLRWM.sorteosSeleccionados.append(self.nombreSorteo)

            print (f"Se seleccion el sorteo {self.nombreSorteo}")

    
    def VariableInfoModel(self):
        print(
        f"mode: {self.mode}, "
        f"X_: {self.X_}, "
        f"y_: {self.y_}, "
        f"fechaCelebra: {self.fechaCelebra}, "
        f"emision: {self.emision}, "
        f"nombreSorteo: {self.nombreSorteo}, "
        f"sorteosEntrenamiento: {self.sorteosEntrenamiento}, "
        f"tipoProducto: {self.tipoProducto}")
        
    def predict(self):
        
         
        if (self.data["NOMBRE"]==self.nombreSorteo).any():
            dfEntrena=self.data.drop(self.data.loc[self.data["NOMBRE"]==self.nombreSorteo].last_valid_index())
        else:
            dfEntrena=self.data


        dfTrain = dfEntrena.loc[dfEntrena["NOMBRE"].isin(self.sorteosEntrenamiento), [self.X_, self.y_, "NOMBRE"]]


        resultados=[]
        mejoresMSE = [(0, 0, 0)] 
        test_sizes = [0.2, 0.19, 0.18,0.17,0.14, 0.21, 0.22]  # Los tamaños de partición que deseas probar
        test_size_index = 0  # Índice para recorrer los tamaños de test_size

        while test_size_index < len(test_sizes) and all(r2score < 0.99 for _, _, r2score in mejoresMSE):
            test_size = test_sizes[test_size_index]
            test_size_index += 1

            for j in range(1,50):
                for i in range(1,50):    
                    df_train_split, df_test_split = train_test_split(dfTrain, test_size=test_size, random_state=j)

                    # Luego de la división:
                    X_train = df_train_split[[self.X_]].values
                    y_train = df_train_split[self.y_].values

                    X_test = df_test_split[[self.X_]].values
                    y_test = df_test_split[self.y_].values
  
                    steps = [
                        ('poly', PolynomialFeatures(degree=i)),
                        ('linear', LinearRegression())
                    ]
                    LinearRegressionPipeline=Pipeline(steps=steps)
                    LinearRegressionPipeline.fit(X_train,y_train)

                    y_pred=LinearRegressionPipeline.predict(X_train)
                    y_predTest=LinearRegressionPipeline.predict(X_test)

                    r2 = r2_score(y_test, y_predTest)

                    resultados.append((j, i, r2))
                    
            resultados.sort(key=lambda x: x[2], reverse=True)
            mejoresMSE = resultados[:3]
        
        df_train_split, df_test_split = train_test_split(dfTrain, test_size=1, random_state=mejoresMSE[0][0])

        # Luego de la división:
        X_train = df_train_split[[self.X_]].values
        y_train = df_train_split[self.y_].values

        X_test = df_test_split[[self.X_]].values
        y_test = df_test_split[self.y_].values
        
        steps = [
            ('poly', PolynomialFeatures(degree=mejoresMSE[0][1])),
            ('linear', LinearRegression())
        ]

        LinearRegressionPipeline=Pipeline(steps=steps)
        LinearRegressionPipeline.fit(X_train,y_train)


        self.df_train_split, self.df_test_split = train_test_split(dfTrain, test_size=1, random_state=mejoresMSE[0][0])

        # Luego de la división:
        X_train = df_train_split[[self.X_]].values
        y_train = df_train_split[self.y_].values

        X_test = df_test_split[[self.X_]].values
        y_test = df_test_split[self.y_].values


        try:
            maxDNAS=int(dfEntrena.loc[dfEntrena["NOMBRE"]==self.nombreSorteo,"DNAS"].max())
        except:
            raise Exception("Aun no hay datos de venta de este sorteo")
        
            
        self.X_toPredict=np.linspace(0,1,maxDNAS)
        self.y_predict=LinearRegressionPipeline.predict(self.X_toPredict.reshape(-1,1))

        DNASColumn=range(maxDNAS,0,-1)
        IDColumn=self.data.loc[self.data["NOMBRE"]==self.nombreSorteo,"ID_SORTEO"].max()
        AvanceEstimColumn= self.y_predict

        if self.data.loc[(self.data["NOMBRE"]==self.nombreSorteo),"CANTIDAD_BOLETOS_MEMBRESIAS"].any():
            dfMembresias=self.data.loc[(self.data["NOMBRE"]==self.nombreSorteo),["CANTIDAD_BOLETOS_MEMBRESIAS", "DNAS"]]

        else:
            dfMembresias=self.data.loc[(self.data["NOMBRE"]==self.sorteosEntrenamiento[-1]),["CANTIDAD_BOLETOS_MEMBRESIAS", "DNAS"]]

        

        
        PrediccionesDict={"ID_SORTEO":IDColumn,"SORTEO":self.nombreSorteo,"DNAS":DNASColumn,"TALONES_ESTIMADOS":AvanceEstimColumn*self.emision}
        dfPredicciones=pd.DataFrame(PrediccionesDict)

        dfPredicciones.loc[dfPredicciones["DNAS"]<=2,"TALONES_ESTIMADOS"]+=0
        dfPredicciones['TALONES_DIARIOS_ESTIMADOS'] = dfPredicciones['TALONES_ESTIMADOS'].diff()
        dfPredicciones.iloc[0,4]=dfPredicciones["TALONES_ESTIMADOS"][0]

        
        dfPredicciones=pd.merge(dfPredicciones,dfMembresias,on="DNAS",how="left").fillna(0)
        dfPredicciones["TALONES_DIARIOS_ESTIMADOS"]=dfPredicciones["CANTIDAD_BOLETOS_MEMBRESIAS"]+dfPredicciones["TALONES_DIARIOS_ESTIMADOS"]
        dfPredicciones["TALONES_ESTIMADOS"]=dfPredicciones.groupby("SORTEO")["TALONES_DIARIOS_ESTIMADOS"].cumsum()
        dfPredicciones=dfPredicciones.drop("CANTIDAD_BOLETOS_MEMBRESIAS",axis=1)

        fechaInicio = self.fechaCelebra - timedelta(days=int(maxDNAS))
        fechaFin = self.fechaCelebra

        # Crear el rango de fechas
        rangoFechas = pd.date_range(start=fechaInicio, end=fechaFin)
        
        dfPredicciones["FECHA_MAPEADA"]=rangoFechas[1:]

        dfPredicciones["FECHAAPOYO"]=(dfPredicciones['FECHA_MAPEADA']- pd.Timestamp('1899-12-30')).dt.days
        dfPredicciones["ID_SORTEO_DIA"]=(dfPredicciones["FECHAAPOYO"].astype(str)+dfPredicciones["ID_SORTEO"].astype(str)).astype(np.int64)
        dfPredicciones=dfPredicciones.drop("FECHAAPOYO",axis=1)
        return dfPredicciones
    
    def viewTrain(self):
        fig,ax=plt.subplots(1,1,figsize=(13,7))
        return sn.scatterplot(x=self.X_,y=self.y_,data=self.df_train_split,ax=ax,hue="NOMBRE"), sn.lineplot(x=self.X_toPredict,y=self.y_predict,ax=ax)
        

    def viewReal(self):
        if (self.data["NOMBRE"]==self.nombreSorteo).any():
            fig,ax=plt.subplots(1,1,figsize=(13,7))
            dfSorteoActual=self.data.drop(self.data.loc[self.data["NOMBRE"]==self.nombreSorteo].last_valid_index())
            XRealSorteoActual=pd.array(dfSorteoActual.loc[dfSorteoActual["NOMBRE"]==self.nombreSorteo,self.X_])
            YRealSorteoActual=pd.array(dfSorteoActual.loc[dfSorteoActual["NOMBRE"]==self.nombreSorteo,"PORCENTAJE_DE_AVANCE_CON_MEMBRE"])
            return sn.scatterplot(x=XRealSorteoActual,y=YRealSorteoActual,ax=ax), sn.lineplot(x=self.X_toPredict,y=self.y_predict,ax=ax)
        else:
            raise ValueError("No hay historico de este sorteo para comparar")

ObjPrediccionesTST219=SorteosTecLRWM(tipoProducto="TST",X_="PORCENTAJE_DNAS",y_="PORCENTAJE_DE_AVANCE_SIN_MEMBRE",data=dfBoletosEscalado,dfInfoSorteos=dfInfo,mode="automatic")
dfPrediccionesTST219=ObjPrediccionesTST219.predict()
ObjPrediccionesTST219.viewTrain()
ObjPrediccionesTST219.viewReal()


ObjPrediccionesSMS32=SorteosTecLRWM(tipoProducto="SMS",X_="PORCENTAJE_DNAS",y_="PORCENTAJE_DE_AVANCE_SIN_MEMBRE",data=dfBoletosEscalado,dfInfoSorteos=dfInfo,mode="automatic")
dfPrediccionesSMS32=ObjPrediccionesSMS32.predict()
ObjPrediccionesSMS32.viewTrain()
ObjPrediccionesSMS32.viewReal()

ObjPrediccionesAVT30=SorteosTecLRWM(tipoProducto="AVT",X_="PORCENTAJE_DNAS",y_="PORCENTAJE_DE_AVANCE_SIN_MEMBRE",data=dfBoletosEscalado,dfInfoSorteos=dfInfo,mode="automatic")
dfPrediccionesAVT30=ObjPrediccionesAVT30.predict()
ObjPrediccionesAVT30.viewTrain()
ObjPrediccionesAVT30.viewReal()


ObjPrediccionesSOE48=SorteosTecLRWM(tipoProducto="SOE",X_="PORCENTAJE_DNAS",y_="PORCENTAJE_DE_AVANCE_SIN_MEMBRE",data=dfBoletosEscalado,dfInfoSorteos=dfInfo,mode="automatic")
dfPrediccionesSOE48=ObjPrediccionesSOE48.predict()
ObjPrediccionesSOE48.viewTrain()
ObjPrediccionesSOE48.viewReal()

ObjPrediccionesDDXV10=SorteosTecLRWM(tipoProducto="DXV",X_="PORCENTAJE_DNAS",y_="PORCENTAJE_DE_AVANCE_SIN_MEMBRE",data=dfBoletosEscalado,dfInfoSorteos=dfInfo,mode="automatic")
dfPrediccionesDDXV10=ObjPrediccionesDDXV10.predict()
ObjPrediccionesDDXV10.viewTrain()
ObjPrediccionesDDXV10.viewReal()


dfBoletosReal=dfBoletos.sort_values(["NOMBRE","FECHAREGISTRO"]).loc[:,["NOMBRE","FECHAREGISTRO","CANTIDAD_BOLETOS","PRECIO_UNITARIO"]]
dfBoletosReal["FECHAREGISTRO"]=pd.to_datetime(dfBoletosReal.loc[:,"FECHAREGISTRO"],format='%Y-%m-%d')
dfBoletosReal["INGRESO"]=dfBoletosReal["PRECIO_UNITARIO"]*dfBoletosReal["CANTIDAD_BOLETOS"]
dfBoletosReal=dfBoletosReal.drop("PRECIO_UNITARIO",axis=1)
dfBoletosReal=dfBoletosReal.groupby(["NOMBRE","FECHAREGISTRO"]).agg(CANTIDAD_BOLETOS=("CANTIDAD_BOLETOS","sum"),INGRESO=("INGRESO","sum")).reset_index()
dfBoletosReal=dfBoletosReal.drop(dfBoletosReal.groupby("NOMBRE")['FECHAREGISTRO'].idxmax())


dfBoletosRealMes=dfBoletosReal.loc[(dfBoletosReal["FECHAREGISTRO"].dt.month==mes)&(dfBoletosReal["FECHAREGISTRO"].dt.year==2025)]
dfBoletosRealMes=dfBoletosRealMes.groupby("NOMBRE").agg(TALONES_MES_REAL=("CANTIDAD_BOLETOS","sum"),INGRESO_MES_REAL=("INGRESO","sum")).reset_index()

dfResumenReal=dfBoletosReal.loc[(dfBoletosReal["FECHAREGISTRO"].dt.year<2025)|((dfBoletosReal["FECHAREGISTRO"].dt.year==2025)&(dfBoletosReal["FECHAREGISTRO"].dt.month<mes))]
dfResumenReal=dfResumenReal.groupby("NOMBRE").agg(TALONES_REALES_ACUMULADOS=("CANTIDAD_BOLETOS","sum"),INGRESO_REAL_ACUMULADO=("INGRESO","sum")).reset_index()
dfResumenReal=pd.merge(dfInfo[["NOMBRE","EMISION","PRECIO_UNITARIO","FECHA_CIERRE"]],dfResumenReal,on="NOMBRE",how="right")
dfResumenReal["PORCENTAJE_REAL_ACUMULADO"]=dfResumenReal["TALONES_REALES_ACUMULADOS"]/dfResumenReal["EMISION"]
dfResumenReal=pd.merge(dfResumenReal,dfBoletosRealMes,on="NOMBRE",how="left")
dfResumenReal["PORCENTAJE_MES_REAL"]=dfResumenReal["TALONES_MES_REAL"]/dfResumenReal["EMISION"]
dfResumenReal=dfResumenReal.fillna(0)


fechaTopConDatosReales=dfBoletosReal["FECHAREGISTRO"].max()
dfPrediccionesAll=pd.concat([dfPrediccionesAVT30,dfPrediccionesSOE48,dfPrediccionesDDXV10,dfPrediccionesSMS32,dfPrediccionesTST219])
dfResumenPrediccion=dfPrediccionesAll.loc[(dfPrediccionesAll["FECHA_MAPEADA"]>fechaTopConDatosReales)&(dfPrediccionesAll["FECHA_MAPEADA"].dt.month==mes)&(dfPrediccionesAll["FECHA_MAPEADA"].dt.year==2025)]
dfResumenPrediccion=dfResumenPrediccion.groupby("SORTEO")["TALONES_DIARIOS_ESTIMADOS"].sum().reset_index(name="TALONES_ESTIMADOS_DIAS_FALTANTES")


dfPrediccionesEtimadoTotal=dfPrediccionesAll.groupby("SORTEO")["TALONES_ESTIMADOS"].max().reset_index().rename({"SORTEO":"NOMBRE","TALONES_ESTIMADOS":"TALONES_ESTIMADOS_SORTEO"},axis=1)

dfResumen=pd.merge(dfResumenReal,dfResumenPrediccion,left_on="NOMBRE",right_on="SORTEO").drop("SORTEO",axis=1)
dfResumen=pd.merge(dfResumen,dfPrediccionesEtimadoTotal, on="NOMBRE",how="left")
dfResumen["INGRESO_ESTIMADO_DIAS_FALTANTES"]=dfResumen["PRECIO_UNITARIO"]*dfResumen["TALONES_ESTIMADOS_DIAS_FALTANTES"]
dfResumen["PORCENTAJE_ESTIMADO_DIAS_FALTANTES"]=dfResumen["TALONES_ESTIMADOS_DIAS_FALTANTES"]/dfResumen["EMISION"]
dfResumen["PORCENTAJE_TOTAL_MES_ESTIMADO"]=dfResumen["PORCENTAJE_ESTIMADO_DIAS_FALTANTES"]+dfResumen["PORCENTAJE_MES_REAL"]
dfResumen["PORCENTAJE_ESTIMADO_TOTAL"]=dfResumen["PORCENTAJE_TOTAL_MES_ESTIMADO"]+dfResumen["PORCENTAJE_REAL_ACUMULADO"]
dfResumen["TALONES_ESTIMADO_TOTAL"]=dfResumen["TALONES_ESTIMADOS_DIAS_FALTANTES"]+dfResumen["TALONES_REALES_ACUMULADOS"]+dfResumen["TALONES_MES_REAL"]
dfResumen=dfResumen.sort_values("FECHA_CIERRE").drop("FECHA_CIERRE",axis=1)
dfResumen=dfResumen[["NOMBRE","EMISION","PRECIO_UNITARIO","TALONES_REALES_ACUMULADOS","INGRESO_REAL_ACUMULADO","PORCENTAJE_REAL_ACUMULADO","TALONES_MES_REAL","INGRESO_MES_REAL","PORCENTAJE_MES_REAL","TALONES_ESTIMADOS_DIAS_FALTANTES"
                    ,"INGRESO_ESTIMADO_DIAS_FALTANTES","PORCENTAJE_ESTIMADO_DIAS_FALTANTES","TALONES_ESTIMADOS_SORTEO","PORCENTAJE_TOTAL_MES_ESTIMADO","PORCENTAJE_ESTIMADO_TOTAL","TALONES_ESTIMADO_TOTAL"]]
dfResumen.to_csv("dfResumenSinMembre.csv",header=True, index=False)
dfPrediccionesAll.to_csv("dfPrediccionesAllFisi.csv",header=True,index=False)

dfResumen