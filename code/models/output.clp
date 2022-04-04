\ENCODING=ISO-8859-1
\Problem name: RescheduleFixture_basic.lp_conflict

Maximize
 obj:
Subject To
 c761:  Denver_Nuggets_Phoenix_Suns_2021m01m01_00.00.00_2021m05m16_00.00.00_2021m03m10_00.00.00#9
        + Phoenix_Suns_Atlanta_Hawks_2021m01m13_00.00.00_2021m03m30_00.00.00_2021m03m10_00.00.00#10
        + Phoenix_Suns_Golden_State_Warriors_2021m01m28_00.00.00_2021m05m16_00.00.00_2021m03m12_00.00.00#18
        + Phoenix_Suns_Indiana_Pacers_2021m01m16_00.00.00_2021m03m13_00.00.00_2021m03m10_00.00.00#19
        + Phoenix_Suns_Golden_State_Warriors_2021m01m15_00.00.00_2021m03m04_00.00.00_2021m03m10_00.00.00#59
        + Phoenix_Suns_Golden_State_Warriors_2021m01m15_00.00.00_2021m03m04_00.00.00_2021m03m12_00.00.00#60
        <= 1
 c1508: Denver_Nuggets_Phoenix_Suns_2021m01m01_00.00.00_2021m05m16_00.00.00_2021m03m10_00.00.00#9
         = 1
 c1511: Phoenix_Suns_Golden_State_Warriors_2021m01m28_00.00.00_2021m05m16_00.00.00_2021m03m12_00.00.00#18
         = 1
\Sum of equality rows in the conflict:
\ sum_eq: Denver_Nuggets_Phoenix_Suns_2021m01m01_00.00.00_2021m05m16_00.00.00_2021m03m10_00.00.00#9
\         + Phoenix_Suns_Golden_State_Warriors_2021m01m28_00.00.00_2021m05m16_00.00.00_2021m03m12_00.00.00#18
\          = 2
Bounds
 0 <= Denver_Nuggets_Phoenix_Suns_2021m01m01_00.00.00_2021m05m16_00.00.00_2021m03m10_00.00.00#9 <= 1
 0 <= Phoenix_Suns_Atlanta_Hawks_2021m01m13_00.00.00_2021m03m30_00.00.00_2021m03m10_00.00.00#10 <= 1
 0 <= Phoenix_Suns_Golden_State_Warriors_2021m01m28_00.00.00_2021m05m16_00.00.00_2021m03m12_00.00.00#18 <= 1
 0 <= Phoenix_Suns_Indiana_Pacers_2021m01m16_00.00.00_2021m03m13_00.00.00_2021m03m10_00.00.00#19 <= 1
 0 <= Phoenix_Suns_Golden_State_Warriors_2021m01m15_00.00.00_2021m03m04_00.00.00_2021m03m10_00.00.00#59 <= 1
 0 <= Phoenix_Suns_Golden_State_Warriors_2021m01m15_00.00.00_2021m03m04_00.00.00_2021m03m12_00.00.00#60 <= 1
Binaries
 Denver_Nuggets_Phoenix_Suns_2021m01m01_00.00.00_2021m05m16_00.00.00_2021m03m10_00.00.00#9 
 Phoenix_Suns_Atlanta_Hawks_2021m01m13_00.00.00_2021m03m30_00.00.00_2021m03m10_00.00.00#10 
 Phoenix_Suns_Golden_State_Warriors_2021m01m28_00.00.00_2021m05m16_00.00.00_2021m03m12_00.00.00#18 
 Phoenix_Suns_Indiana_Pacers_2021m01m16_00.00.00_2021m03m13_00.00.00_2021m03m10_00.00.00#19 
 Phoenix_Suns_Golden_State_Warriors_2021m01m15_00.00.00_2021m03m04_00.00.00_2021m03m10_00.00.00#59 
 Phoenix_Suns_Golden_State_Warriors_2021m01m15_00.00.00_2021m03m04_00.00.00_2021m03m12_00.00.00#60 
End
