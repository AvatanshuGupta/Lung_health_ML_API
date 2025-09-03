from pydantic import BaseModel,Field
from typing import Literal
from enum import Enum
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 309 entries, 0 to 308
Data columns (total 14 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   GENDER                 309 non-null    int64  
 1   AGE                    309 non-null    float64
 2   SMOKING                309 non-null    int64  
 3   YELLOW_FINGERS         309 non-null    int64  
 4   ANXIETY                309 non-null    int64  
 5   FATIGUE                309 non-null    int64  
 6   ALLERGY                309 non-null    int64  
 7   WHEEZING               309 non-null    float64
 8   ALCOHOL CONSUMING      309 non-null    int64  
 9   COUGHING               309 non-null    int64  
 10  SHORTNESS OF BREATH    309 non-null    int64  
 11  SWALLOWING DIFFICULTY  309 non-null    int64  
 12  CHEST PAIN             309 non-null    int64  
 13  LUNG_CANCER            309 non-null    int64  
dtypes: float64(2), int64(12)
"""
class WheezingEnum(float, Enum):
    NO = 1.0
    YES = 2.0

class Patient(BaseModel):
    GENDER: Literal[0, 1]
    AGE: float
    SMOKING: Literal[1, 2]
    YELLOW_FINGERS: Literal[1, 2]
    ANXIETY: Literal[1, 2]
    FATIGUE: Literal[1, 2]
    ALLERGY: Literal[1, 2]
    WHEEZING: WheezingEnum
    ALCOHOL_CONSUMING: Literal[1, 2]
    COUGHING: Literal[1, 2]
    SHORTNESS_OF_BREATH: Literal[1, 2]
    SWALLOWING_DIFFICULTY: Literal[1, 2]
    CHEST_PAIN: Literal[1, 2]
