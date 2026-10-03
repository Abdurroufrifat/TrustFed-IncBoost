"""Rebuild the two methodology figures as editable, vector PDF diagrams."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
INK = '#20364B'
MUTED = '#51677A'
BLUE = '#DCEAF4'
TEAL = '#DDF2ED'
GOLD = '#FFF0D8'
PURPLE = '#EDEAF6'
GRAY = '#F1F4F6'


def canvas(h):
    fig, ax = plt.subplots(figsize=(12.0, h))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 12), ylim=(0, h))
    ax.axis('off')
    fig.subplots_adjust(left=.02, right=.98, bottom=.035, top=.96)
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), fill=BLUE, size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.035,rounding_size=0.13',
        facecolor=fill, edgecolor=INK, linewidth=1.15))
    ax.text(x+w/2, y+h-.25, title, ha='center', va='top', color=INK,
            weight='bold', fontsize=size)
    for j, line in enumerate(lines):
        ax.text(x+w/2, y+h-.66-j*.27, line, ha='center', va='top',
                color=MUTED, fontsize=8.45)


def arrow(ax, start, end, color=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
        mutation_scale=13, linewidth=1.45, color=color,
        connectionstyle='arc3,rad=0'))


def label(ax, x, y, txt, size=8.5, color=MUTED, **kwargs):
    ax.text(x, y, txt, fontsize=size, color=color, **kwargs)


fig, ax = canvas(4.65)
label(ax, .3, 4.33, '01  |  CAG-FE: FEDERATED DETECTOR', 13, INK, weight='bold', va='center')
label(ax, 11.75, 4.33, 'WUSTL-EHMS', 8.2, MUTED, ha='right', va='center')
box(ax, .30, 2.09, 2.0, 1.60, 'Five clients',
    ('Private local records', 'Local XGBoost models', 'Predictions shared'), BLUE)
box(ax, 2.68, 2.09, 2.0, 1.60, 'Clean validation',
    ('Server-held reference', 'F1, balanced accuracy', 'ROC-AUC; quality'), GRAY)
box(ax, 5.06, 2.09, 2.0, 1.60, 'Chance-aware gate',
    ('Below chance margin?', 'Reject client; weight = 0', 'Admitted clients only'), GOLD)
box(ax, 7.44, 2.09, 2.0, 1.60, 'Trust and cap',
    ('Quality + calibration', 'Bounded agreement', 'Each weight ≤ 0.40'), TEAL)
box(ax, 9.82, 2.09, 1.85, 1.60, 'Final detector',
    ('Weighted predictions', 'Validation-set threshold', 'Held-out test'), PURPLE)
for a,b in [(2.32,2.64),(4.70,5.02),(7.08,7.40),(9.46,9.78)]:
    arrow(ax,(a,2.89),(b,2.89))
ax.plot([.35,11.65],[1.72,1.72], color='#D9E1E8', lw=1)
label(ax,.48,1.40,'CLIENT BOUNDARY',8.4,INK,weight='bold')
label(ax,.48,1.05,'Local records stay at each simulated client.',9.2)
label(ax,6.14,1.40,'SERVER DECISION',8.4,INK,weight='bold')
label(ax,6.14,1.05,'Admission precedes trust scoring and capped aggregation.',9.2)
fig.savefig(OUT/'system_overview.pdf', bbox_inches='tight', pad_inches=.08)
fig.savefig(OUT/'system_overview.png', dpi=190, bbox_inches='tight', pad_inches=.08)
plt.close(fig)

fig, ax = canvas(5.05)
label(ax,.3,4.75,'02  |  PHYSIOLOGICAL STATE PROTECTION',13,INK,weight='bold',va='center')
label(ax,11.7,4.75,'PHYSIONET / CINC 2015',8.2,MUTED,ha='right',va='center')
box(ax,.35,3.57,2.40,.83,'627 ECG + PLETH recordings',(),BLUE,9.1)
box(ax,3.23,3.57,2.45,.83,'Split by recording',('376 train · 125 val · 126 test',),GRAY,9.2)
box(ax,6.15,3.57,2.46,.83,'Controlled fault windows',('Gain · dropout · flatline · replay',),GOLD,9.2)
for start,end in [((2.78,3.98),(3.18,3.98)),((5.71,3.98),(6.10,3.98))]: arrow(ax,start,end)
label(ax,.38,3.24,'MODEL TRAINING   •   SIMULATED CLIENTS',8.3,INK,weight='bold')
box(ax,.35,2.12,2.40,.88,'Five local XGBoost models',('Training recordings only',),TEAL,9.25)
box(ax,3.23,2.12,2.45,.88,'Trust-weighted federation',('Validation-selected weights',),TEAL,9.0)
box(ax,6.15,2.12,2.46,.88,'Frozen fault detector',('Scores held-out windows',),TEAL,9.2)
arrow(ax,(2.78,2.56),(3.18,2.56));arrow(ax,(5.71,2.56),(6.10,2.56))
label(ax,.38,1.78,'CAUSAL STATE UPDATE   •   EACH HELD-OUT RECORDING',8.3,INK,weight='bold')
box(ax,.35,.49,2.40,1.02,'Clean initial reference',('First 30 seconds assumed clean',),PURPLE,9.1)
box(ax,3.23,.49,2.45,1.02,'Observed 5 s window',('Compare with stored history',),PURPLE,9.15)
box(ax,6.15,.49,2.46,1.02,'Update gate',('Deviation + fault + replay checks',),GOLD,9.2)
box(ax,9.05,.49,2.60,1.02,'Recording-specific state',('Accept update or keep prior state',),BLUE,9.0)
for start,end in [((2.78,1.0),(3.18,1.0)),((5.71,1.0),(6.10,1.0)),((8.64,1.0),(9.0,1.0))]:arrow(ax,start,end)
arrow(ax,(7.38,2.07),(7.38,1.56))
label(ax,11.66,2.95,'Recorded waveforms; simulated clients and faults.',7.7,MUTED,ha='right')
fig.savefig(OUT/'physionet_methodology.pdf', bbox_inches='tight', pad_inches=.08)
fig.savefig(OUT/'physionet_methodology.png', dpi=190, bbox_inches='tight', pad_inches=.08)
plt.close(fig)
