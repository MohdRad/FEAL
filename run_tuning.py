from FEAL import tuning, tuning_visual
from stratified import tuning_strata 

alpha = [0.001, 0.01, 0.1, 1, 2, 3]
#=======================================
# uncertainty sampling
# Assoc FE
tuning(alpha, 'assoc', 42)

# Diassoc FE 
tuning(alpha, 'disassoc', 42)

# Plot
tuning_visual(data_path= './cases/tuning/alpha_AL_assoc.csv', 
              name = 'assoc', 
              xlim= [-0.1,3.1], 
              ylim = [0.91,0.944],
              xticks = [0, 1, 2, 3], 
              yticks = [0.91, 0.92, 0.93, 0.94],
              title='(a)')

tuning_visual(data_path= './cases/tuning/alpha_AL_disassoc.csv', 
              name = 'disassoc', 
              xlim= [-0.1,3], 
              ylim = [0.79,0.872],
              xticks = [0, 1, 2, 3.1], 
              yticks = [0.76, 0.79, 0.82, 0.85, 0.88],
              title='(b)')

#=======================================
# Stratified sampling 
# Assoc FE
#tuning_strata(alpha, 'assoc', 42)

# Disassoc FE
#tuning_strata(alpha, 'disassoc', 42)


