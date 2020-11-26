[data]

########define the UIRT columns
USER_ID_FIELD='session_id'
ITEM_ID_FIELD='item_id'
NEG_PREFIX='neg_'
LABEL_FIELD='label'
RATING_FIELD='rating'

#########select load columns
# USER_ID_FIELD & ITEM_ID_FIELD can be omitted
load_col={'inter': ['user_id', 'item_id', 'timestamp']}

########data separator
field_separator='\t'
seq_separator=' '

########data filter
#lowest_val={'user_id':0}
min_user_inter_num = 2

########label threshold
# threshold={'rating':3}
