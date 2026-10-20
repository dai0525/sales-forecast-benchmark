"""Rossmann covariates with conservative historical schedule features."""
from datetime import date
import numpy as np
import pandas as pd
from bench_rossmann import Panel,PANEL,ROOT
GROUPS={'promo':[0],'school':[1],'state_holiday':[2,3,4],'promo2_active':[5],'competition_open':[6]}
NAMES=['Promo','SchoolHoliday','StateHoliday_a','StateHoliday_b','StateHoliday_c','Promo2Active','CompetitionOpen']


class FactorPanel(Panel):
    def __init__(self, panel_path=PANEL):
        super().__init__(panel_path)
        store=pd.read_csv(ROOT/'data/alt/rossmann/store.csv').set_index('Store').loc[self.stores]
        self.promo2_start=np.full(self.n,np.datetime64('2100-01-01'),dtype='datetime64[D]')
        self.competition_start=self.promo2_start.copy();self.promo2_months=np.zeros((self.n,12),bool)
        months={'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sept':9,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
        for i,(_,r) in enumerate(store.iterrows()):
            if r.Promo2==1 and pd.notna(r.Promo2SinceYear) and pd.notna(r.Promo2SinceWeek):
                self.promo2_start[i]=np.datetime64(date.fromisocalendar(int(r.Promo2SinceYear),int(r.Promo2SinceWeek),1))
                if pd.notna(r.PromoInterval):
                    for month in r.PromoInterval.split(','):self.promo2_months[i,months[month]-1]=True
            if pd.notna(r.CompetitionOpenSinceYear) and pd.notna(r.CompetitionOpenSinceMonth):
                self.competition_start[i]=np.datetime64(date(int(r.CompetitionOpenSinceYear),int(r.CompetitionOpenSinceMonth),1))
        self.extra=np.zeros((self.n,self.T,7),float)
        self.extra[:,:,0]=self.promo;self.extra[:,:,1]=self.school
        for j in range(3):self.extra[:,:,2+j]=(self.state==j+1)
        self.extra[:,:,5]=(self.days[None,:]>=self.promo2_start[:,None])*self.promo2_months[:,self.cal['month']-1]
        self.extra[:,:,6]=(self.days[None,:]>=self.competition_start[:,None])

    def at_origin(self,o,h):
        hist=self.extra[:,:o].copy();future=self.extra[:,o:o+h].copy()
        future[:,:,5]*=(self.promo2_start<=self.days[o-1])[:,None]
        future[:,:,6]=(self.competition_start<=self.days[o-1])[:,None]
        return hist,future
