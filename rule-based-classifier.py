import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Load your data
df = pd.read_csv("Megan.csv")
df.columns = df.columns.str.strip()
df['final code'] = pd.to_numeric(df['Code'], errors='coerce')

# Enhanced keywords with regex patterns
keyword_patterns = {
    "2a_gendered_slur": [
        r'\bslut+[syz]?\b',  # slut, sluts, slutz, slutty
        r'\bwhor[eaio]+[sd]?\b',  # whore, whores, whoring
        r'\bc+u+n+t+s?\b',  # cunt, cunts (with letter repetition)
        r'\bb+i+t+c+h+[eyi]*[sz]?\b',  # bitch, bitches, bitchy
        r'\btwat+s?\b',
        r'\bh[o0]+e+s?\b',  # hoe, hoes
        r'\bth[o0]t+s?\b',  # thot, thots
        r'\bskank+[syz]?\b',
        r'\bslag+s?\b',
        r'\btart+s?\b',
        r'\bhussy\b',
        r'\btrollop\b',
        r'\bstrumpet\b',
        r'\bfl[o0]+zy\b',
        r'\bharlot+s?\b',
        r'\btramp+s?\b',
        r'\bharpy\b',
        r'\bvixen\b',
        r'\bhellcat\b',
        r'\bslattern\b',
        r'\battention\s+wh[o0]re\b',
        r'\btwit+s?\b',
    ],
    "2b_sexual_attack": [
        r'\bprostit\w+',  # prostitute, prostitutes, prostitution
        r'\blo+se?\s+wom[ae]n\b',
        r'\bstreet\s+girl\b',
        r'\bbawd\b',
        r'\bcrack\s*wh[o0]re\b',
        r'\bcrack\s*h[o0]\b',
        r'\bturn(ing)?\s+tricks\b',
        r'\bc[o0]ck\s*m[o0]nger\b',
        r'\bcum\s*bag\b',
        r'\bbend\s+over\b',
        r'\brap[eaio]+[ds]?\b',  
        r'\bsex(ual)?\s+object\b',
        r'\brevenge\s+porn\b',
        r'\bsend\s+nudes?\b',
    ],
    "2c_gendered_intellectual_inferiority": [
        r'\bdumb\s+(twat|cunt|bitch|woman|girl)\b',
        r'\bstupid\s+(woman|girl)\b',
        r'\btypical\s+wom[ae]n\b',
        r'\bt[o0]+\s+em[o0]ti[o0]nal\b',
        r'\bwom[ae]n\s+can+[\'t]+\b',
        r'\bgirls?\s+can+[\'t]+\b',
        r'\bgirl\s+you\s+are\s+dumb\b',
        r'\bwom[ae]n\s+are\s+dumb\b',
    ],
    "2d_call_for_action_violence": [
        r'\bkill\s+her\b',
        r'\bshoot\s+her\b',
        r'\bhurt\s+her\b',
        r'\bthrow\s+a\s+bomb\s+at\s+you\b',
        r'\block\s+her\s+up\b',
        r'\bbitch[\-\s]*slap\b',
    ],
    "3b_assumed_incompetence": [
        r'\b(go\s+)?back\s+to\s+(the\s+)?bartend(ing|er)?\b',
        r'\bbar\s*keep\b',
        r'\bmake\s+me\s+a\s+(drink|margarita|cocktail|sandwich)\b',
        r'\bback\s+to\s+(the\s+)?kitchen\b',
        r'\bd[o0]\s+n[o0]t\s+bel[o0]ng\s+in\b',
        r'\bw[o0]rking\s+girl\b',
        r'\bserve\s+(them\s+)?drinks?\b',
        r'\b[o0]nly\s+(thing\s+)?you\s+are\s+qualified\b',
        r'\biq\s+[o0]f\s+a\s+brick\b',
        r'\bdumbest\s+member\b',
        r'\bdumber\s+than\s+dumb\b',
        r'\byou\s+s[o0]\s+dumb\b',
        r'\bgirl\s+u\s+are\s+dumb\b',
        r'\bcompl[e3]te\s+dumbass\b',
        r'\bwhat\s+an?\s+idi[o0]t\b',
        r'\byou\s+are\s+(clueless|such\s+a?\s+buff[o0]+n)\b',
        r'\bhigh\s+iq\b',
        r'\bm[o0]r[o0]n+s?\b',
        r'\bidi[o0]t+s?\b',
        r'\bdumbass+\b',
        r'\bbuff[o0]+n+s?\b',
        r'\bph[o0]n[eyi]+\b',
        r'\blady\s+naive\b',
        r'\bretard+[se]?\b',
        r'\bdelusi[o0]nal\b',
        
    ],
    "3c_infantilizing": [
        r'\bsweetheart\b',
        r'\bmissy\b',
        r'\blittle\s+lady\b',
        r'\bcalm\s+down\s+sweetie\b',
        r'\bg[o0]+d\s+girl\b',
        r'\bhun\s',  # with space to avoid "hundred"
        r'\bdoll\b',
        r'\bchick+s?\b',
        r'\bbabe+s?\b',
        r'\bbird+s?\b',
        r'\bdame+s?\b',
        r'\bpuss\b',
        r'\bget\s+l[o0]st\s+lady\b',
        r'\bl[o0]st\s+lady\b',
    ],
    "3e_demonstration_of_power": [
        r'\bstay\s+clear\s+[o0]f\s+you\b',
        r'\byour\s+p[o0]is[o0]n\b',
        r'\bhandlers\b',
        r'\byour\s+script\b',
    ],
    "3g_appearance_attack": [
        r'\bugly+\b',
        r'\bfea\b',
        r'\bhorse\s*teeth?\b',
        r'\blipstick\s+[o0]n\s+(a\s+)?pig\b',
        r'\bc[o0]w+s?\b',
        r'\balley\s+cat\b',
        r'\bd[o0]nkey+s?\b',
        r'\bl[o0]+k\s+pretty\b',
        r'\bpay\s+t[o0]\s+l[o0]+k\b',
        r'\bwhich\s+animal\s+are\s+you\b',
    ]
}

# Compile all patterns for efficiency
compiled_patterns = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in keyword_patterns.items()
}

# Gender markers and degrading words for combo detection
gender_markers = [
    r'\bgirls?\b', r'\bwom[ae]n\b', r'\bfemales?\b',
    r'\bladies\b', r'\blady\b', r'\bshe\b', r'\bher\b',
    r'\ba[o0]c\b', r'\balexandria\b'
]

degrading_words = [
    r'\bdumb\b', r'\bstupid\b', r'\bidi[o0]t\b', r'\bm[o0]r[o0]n\b',
    r'\bretard\b', r'\bclueless\b', r'\binc[o0]mpetent\b',
    r'\bairhead\b', r'\bbrainless\b', r'\bdimwit\b',
    r'\bimbecile\b', r'\bign[o0]rant\b', r'\bdumbass\b',
    r'\bbuff[o0]+n\b', r'\bf[o0]+l\b', r'\bdummy\b'
]

# Compile gender and degrading patterns
compiled_gender_markers = [re.compile(pattern, re.IGNORECASE) for pattern in gender_markers]
compiled_degrading_words = [re.compile(pattern, re.IGNORECASE) for pattern in degrading_words]

def normalize_text(text):
    """Basic normalization while preserving structure for regex"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Less aggressive normalization to preserve word boundaries
    return text

def check_gender_degrading_combo(reply_text):
    """Check for gender marker + degrading word within proximity"""
    if not isinstance(reply_text, str):
        return False, None
    
    normalized = normalize_text(reply_text)
    
    # Find all gender marker positions
    for gender_pattern in compiled_gender_markers:
        for gender_match in gender_pattern.finditer(normalized):
            gender_pos = gender_match.start()
            gender_word = gender_match.group()
            
            # Check for degrading words within ±50 characters
            window_start = max(0, gender_pos - 50)
            window_end = min(len(normalized), gender_pos + 50)
            window = normalized[window_start:window_end]
            
            for degrading_pattern in compiled_degrading_words:
                if degrading_pattern.search(window):
                    degrading_word = degrading_pattern.search(window).group()
                    return True, f"2c_gender_combo: {gender_word} + {degrading_word}"
    
    return False, None

def classify_reply(reply_text):
    """Classify reply using regex pattern matching"""
    if not isinstance(reply_text, str):
        return "non-misogynistic", None
    
    normalized = normalize_text(reply_text)
    
    # Check keyword patterns
    for category, patterns in compiled_patterns.items():
        for pattern in patterns:
            match = pattern.search(normalized)
            if match:
                return "misogynistic", f"{category} (matched: '{match.group()}')"
    
    # Check gender marker + degrading word combo
    is_combo, combo_category = check_gender_degrading_combo(reply_text)
    if is_combo:
        return "misogynistic", combo_category
    
    return "non-misogynistic", None

# Apply classification
df["classifier_label"], df["matched_category"] = zip(
    *df["Reply"].apply(classify_reply)
)

# Evaluation
# actual = df['final code'].apply(lambda x: 1 if x in [2.0, 3.0] else 0)
# predicted = (df['classifier_label'] == 'misogynistic').astype(int)

# precision = precision_score(actual, predicted, zero_division=0)
# recall = recall_score(actual, predicted, zero_division=0)
# f1 = f1_score(actual, predicted, zero_division=0)

# false_negatives = df[
#     (df['final code'].isin([2, 3])) &
#     (df['classifier_label'] == 'non-misogynistic')
# ]
# false_positives = df[
#     (~df['final code'].isin([2, 3])) &
#     (df['classifier_label'] == 'misogynistic')
# ]
# true_positives = df[
#     (df['final code'].isin([2, 3])) &
#     (df['classifier_label'] == 'misogynistic')
# ]

# print(f"Total misogynistic replies: {len(df[df['final code'].isin([2,3])])}")
# print(f"True positives  (correctly caught): {len(true_positives)}")
# print(f"False negatives (missed):           {len(false_negatives)}")
# print(f"False positives (wrongly flagged):  {len(false_positives)}")
# print(f"\nPrecision: {precision:.2f}")
# print(f"Recall:    {recall:.2f}")
# print(f"F1:        {f1:.2f}")

# print("\n--- MISSED REPLIES ---")
# for _, row in false_negatives.iterrows():
#     print(f"\nCode {row['final code']}: {row['Reply']}")
#     print(f"Why misogynistic: {row['justification.1']}")

# print("\n--- WRONGLY FLAGGED ---")
# for _, row in false_positives.iterrows():
#     print(f"\nActual code {row['final code']}: {row['Reply']}")
#     print(f"Matched category: {row['matched_category']}")

# print(f"\n--- CORRECTLY CAUGHT REPLIES ({len(true_positives)}) ---")
# for _, row in true_positives.iterrows():
#     print(f"\nCode {row['final code']}: {row['Reply']}")
#     print(f"Matched category: {row['matched_category']}")
#     print(f"Why misogynistic: {row['justification.1']}")

# --- STEP 1: Convert labels to binary (misogyny vs non) ---

def is_misogynistic_label(label):
    if pd.isna(label):
        return 0
    parts = str(label).replace(';', ',').split(',')
    if any(p.strip().startswith('2') for p in parts):
        return 2
    if any(p.strip().startswith('3') for p in parts):
        return 3
    return 0

# Ground truth (actual)
df['actual_binary'] = df['Code'].apply(is_misogynistic_label)

# Predictions
df['predicted_binary'] = (df['classifier_label'] == 'misogynistic').astype(int)


# --- STEP 2: Confusion matrix counts ---

tp = ((df['actual_binary'] != 0) & (df['predicted_binary'] == 1)).sum()
fn = ((df['actual_binary'] != 0) & (df['predicted_binary'] == 0)).sum()
fp = ((df['actual_binary'] == 0) & (df['predicted_binary'] == 1)).sum()
tn = ((df['actual_binary'] == 0) & (df['predicted_binary'] == 0)).sum()


# --- STEP 3: Core metrics ---

total = len(df)
actual_misogyny = (df['actual_binary'] != 0).sum()

percent_misogyny = (actual_misogyny / total) * 100 if total > 0 else 0

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0


# --- STEP 4: Error rates ---

false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

percent_fp_total = (fp / total) * 100 if total > 0 else 0
percent_fn_total = (fn / total) * 100 if total > 0 else 0


# --- STEP 5: Print results ---

print(f"Total replies: {total}")
print(f"Actual misogynistic replies: {actual_misogyny} ({percent_misogyny:.2f}%)")

print("\n--- CONFUSION MATRIX ---")
print(f"True Positives:  {tp}")
print(f"False Negatives: {fn}")
print(f"False Positives: {fp}")
print(f"True Negatives:  {tn}")

print("\n--- MODEL PERFORMANCE ---")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")

print("\n--- ERROR ANALYSIS ---")
print(f"False Positive Rate (FPR): {false_positive_rate:.2f}")
print(f"False Negative Rate (FNR): {false_negative_rate:.2f}")
print(f"False Positives (% of total): {percent_fp_total:.2f}%")
print(f"False Negatives (% of total): {percent_fn_total:.2f}%")
explicit = df['Code'].apply(lambda x: any(p.strip().startswith('2') for p in str(x).replace(';',',').split(','))).sum()
implicit = df['Code'].apply(lambda x: any(p.strip().startswith('3') for p in str(x).replace(';',',').split(',')) and not any(p.strip().startswith('2') for p in str(x).replace(';',',').split(','))).sum()
print(f"\n--- MISOGYNY TYPE BREAKDOWN ---")
print(f"Explicit (code 2): {explicit} ({(explicit/total)*100:.2f}%)")
print(f"Implicit (code 3): {implicit} ({(implicit/total)*100:.2f}%)")


# --- STEP 6: Slice results ---

false_positives_df = df[
    (df['actual_binary'] == 0) & (df['predicted_binary'] == 1)
]

false_negatives_df = df[
    (df['actual_binary'] != 0) & (df['predicted_binary'] == 0)
]

true_positives_df = df[
    (df['actual_binary'] != 0) & (df['predicted_binary'] == 1)
]

true_negatives_df = df[
    (df['actual_binary'] == 0) & (df['predicted_binary'] == 0)
]


print("\n--- FALSE NEGATIVES (missed misogyny) ---")
for _, row in false_negatives_df.iterrows():
    print(f"\nReply: {row['Reply']}")
    print(f"Actual label: {row['Code']}")
    print(f"Why misogynistic: {row.get('Justification', 'N/A')}")


print("\n--- FALSE POSITIVES (wrongly flagged) ---")
for _, row in false_positives_df.iterrows():
    print(f"\nReply: {row['Reply']}")
    print(f"Actual label: {row['Code']}")
    print(f"Matched category: {row['matched_category']}")

print(f"\n--- TRUE POSITIVES ({len(true_positives_df)}) ---")
for _, row in true_positives_df.iterrows():
    print(f"\nReply: {row['Reply']}")
    print(f"Matched category: {row['matched_category']}")
    print(f"Actual label: {row['Code']}")

false_negatives_df.to_csv("false_negatives.csv", index=False)
false_positives_df.to_csv("false_positives.csv", index=False)
true_positives_df.to_csv("true_positives.csv", index=False)