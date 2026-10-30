#!/usr/bin/env python3
"""Render the equation-based architecture diagram without rerunning simulations."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
COLORS=['#315A7D']
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
def savefig(name):
 plt.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.035)
 plt.savefig(ROOT/'figures'/f'{name}.png',bbox_inches='tight',pad_inches=.035,dpi=220)
 plt.close()
def render():
  fig,ax=plt.subplots(figsize=(7,3.5));ax.axis('off');ax.set_xlim(0,10.2);ax.set_ylim(.35,5.6)
  nodes={'X':(1.5,3.5,'Passenger activity\n$X$'),'Y':(5.0,3.5,'Annual emissions\n$Y$'),
   'Z':(8.5,3.5,'Technology\nadoption $Z$'),'B':(1.5,1.1,'Demand\nintervention $\\beta$'),
   'P':(6.8,1.1,'Policy support\n$P$'),'S':(9.1,1.1,'Supply capacity\n$S$')}
  for key,(x,y,label) in nodes.items():
   ax.add_patch(FancyBboxPatch((x-.95,y-.48),1.90,.96,boxstyle='round,pad=.06',facecolor='#EAF1F4' if key!='Z' else '#E4F1E9',edgecolor=COLORS[0],lw=1))
   ax.text(x,y,label,ha='center',va='center',fontsize=8.4)
  def arrow(pa,pb,rad=0):
   ar=FancyArrowPatch(pa,pb,arrowstyle='-|>',mutation_scale=10,connectionstyle=f'arc3,rad={rad}',lw=1,color='#5D6F7B')
   ax.add_patch(ar)
  arrow((2.51,3.74),(3.98,3.74));ax.text(3.25,4.06,'activity source',ha='center',fontsize=7.4)
  arrow((3.98,3.24),(2.51,3.24));ax.text(3.25,2.87,'cancellation',ha='center',fontsize=7.4)
  arrow((7.48,3.5),(6.02,3.5));ax.text(6.75,3.84,'abatement',ha='center',fontsize=7.4)
  arrow((8.5,4.05),(1.5,4.05),rad=.25);ax.text(5,5.14,'rebound channel',ha='center',fontsize=8)
  arrow((1.5,1.66),(1.5,2.94));arrow((6.8,1.66),(8.1,2.94));arrow((9.1,1.66),(8.8,2.94))
  savefig('figure1')
 

if __name__=="__main__":render()
