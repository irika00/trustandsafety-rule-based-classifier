import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import itertools

df = pd.read_csv("celebrities.csv")
df.columns = df.columns.str.strip()
df['final code'] = pd.to_numeric(df['Code'], errors='coerce')



#normalization
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    leet_map = {
        '0': 'o', '1': 'i', '3': 'e', '@': 'a',
        '$': 's', '!': 'i', '4': 'a', '5': 's',
    }
    for k, v in leet_map.items():
        text = text.replace(k, v)
    return text

#keyword patterns

keyword_patterns = {
    "2a_gendered_slur": [
        r'\bslut+[syz]?\b',
        r'\bwhor[eaio]+[sd]?\b',
        r'\bc+u+n+t+s?\b',
        r'\bb+i+t+c+h+[eyi]*[sz]?\b',
        r'\btwat+s?\b',
        r'\bhoe+s?\b',
        r'\bthot+s?\b',
        r'\bskank+[syz]?\b',
        r'\bslag+s?\b',
        r'\btart+s?\b',
        r'\bhussy\b',
        r'\btrollop\b',
        r'\bstrumpet\b',
        r'\bflozy\b',
        r'\bharlot+s?\b',
        r'\btramp+s?\b',
        r'\bharpy\b',
        r'\bvixen\b',
        r'\bhellcat\b',
        r'\bslattern\b',
        r'\battention\s+whore\b',
        r'\btwit+s?\b',
        r'\bwench\b',
        r'\bshrew\b',
        r'\bnag\b',
        r'\bfemale\s+dog\b',
        r'\bcow\b',
    ],
    "2b_sexual_attack": [
        r'\bprostit\w+',
        r'\blose\s+women\b',
        r'\bstreet\s+girl\b',
        r'\bbawd\b',
        r'\bcrack\s*whore\b',
        r'\bcrack\s*ho\b',
        r'\bturning\s+tricks\b',
        r'\bcock\s*monger\b',
        r'\bcum\s*bag\b',
        r'\bbend\s+over\b',
        r'\brape[ds]?\b',
        r'\bsexual\s+object\b',
        r'\brevenge\s+porn\b',
        r'\bsend\s+nudes?\b',
        r'\bsex\s+tape\b',
        r'\bonly\s*fans?\b',
        r'\bsleep\s+your\s+way\b',
        r'\bslept\s+(her\s+)?way\s+to\b',
        r'\buse\s+your\s+body\b',
        r'\bshow\s+some\s+skin\b',
    ],
    "2c_gendered_intellectual_inferiority": [
        r'\bdumb\s+(twat|cunt|bitch|woman|girl)\b',
        r'\bstupid\s+(woman|girl)\b',
        r'\btypical\s+wom[ae]n\b',
        r'\btoo\s+emotional\b',
        r'\bwomen\s+cant\b',
        r'\bgirls?\s+cant\b',
        r'\bgirl\s+you\s+are\s+dumb\b',
        r'\bwomen\s+are\s+dumb\b',
        r'\bwomen\s+don[t\']+\s+belong\b',
        r'\btoo\s+hysterical\b',
        r'\bjust\s+a\s+woman\b',
        r'\bonly\s+a\s+girl\b',
        r'\bwomen\s+should\s+(stay|be\s+quiet|not)\b',
        r'\bfemale\s+brain\b',
    ],
    "2d_call_for_action_violence": [
        r'\bkill\s+her\b',
        r'\bshoot\s+her\b',
        r'\bhurt\s+her\b',
        r'\bthrow\s+a\s+bomb\b',
        r'\block\s+her\s+up\b',
        r'\bbitch[\-\s]*slap\b',
        r'\bpunch\s+her\b',
        r'\bbeat\s+her\b',
        r'\bstab\s+her\b',
        r'\bhang\s+her\b',
        r'\bdrown\s+her\b',
        r'\bwatch\s+your\s+back\b',
        r'\byou[\'ll]+\s+regret\b',
        r'\bcoming\s+for\s+you\b',
        r'\bfind\s+you\b',
        r'\bi\s+know\s+where\s+you\s+(live|are)\b',
    ],
    "3b_assumed_incompetence": [
        r'\bback\s+to\s+(the\s+)?bartend(ing|er)?\b',
        r'\bbarkeep\b',
        r'\bmake\s+me\s+a\s+(drink|margarita|cocktail|sandwich)\b',
        r'\bback\s+to\s+(the\s+)?kitchen\b',
        r'\bdo\s+not\s+belong\s+in\b',
        r'\bworking\s+girl\b',
        r'\bserve\s+(them\s+)?drinks?\b',
        r'\bonly\s+(thing\s+)?you\s+are\s+qualified\b',
        r'\biq\s+of\s+a\s+brick\b',
        r'\bdumbest\s+member\b',
        r'\bdumber\s+than\s+dumb\b',
        r'\byou\s+so\s+dumb\b',
        r'\bgirl\s+u\s+are\s+dumb\b',
        r'\bcomplete\s+dumbass\b',
        r'\bwhat\s+an?\s+idiot\b',
        r'\byou\s+are\s+(clueless|such\s+a?\s+buffoon)\b',
        r'\bhigh\s+iq\b',
        r'\bmoron+s?\b',
        r'\bidiot+s?\b',
        r'\bdumbass+\b',
        r'\bbuffoon+s?\b',
        r'\bphony\b',
        r'\blady\s+naive\b',
        r'\bretard+[se]?\b',
        r'\bdelusional\b',

        r'\bstick\s+to\s+(cooking|cleaning|your\s+lane)\b',
        r'\blet\s+the\s+men\s+(talk|handle|decide)\b',
        r'\bknow\s+your\s+place\b',
        r'\bover\s+your\s+head\b',
        r'\boutside\s+your\s+pay\s+grade\b',
        r'\bnot\s+cut\s+out\s+for\b',
        r'\bshouldn[\'t]+\s+be\s+(in\s+)?politics\b',
        r'\bleave\s+it\s+to\s+(the\s+)?men\b',
        r'\bnot\s+qualified\b',
        r'\bout\s+of\s+your\s+depth\b',
    ],
    "3c_infantilizing": [
        r'\bsweetheart\b',
        r'\bmissy\b',
        r'\blittle\s+lady\b',
        r'\bcalm\s+down\s+sweetie\b',
        r'\bgood\s+girl\b',
        r'\bhun\b',
        r'\bdoll\b',
        r'\bchick+s?\b',
        r'\bbabe+s?\b',
        r'\bbird+s?\b',
        r'\bdame+s?\b',
        r'\bpuss\b',
        r'\bget\s+lost\s+lady\b',
        r'\blost\s+lady\b',

        r'\bpumpkin\b',
        r'\bhoney\b',
        r'\bprincess\b',
        r'\bdarling\b',
        r'\bprecious\b',
        r'\bnot\s+ready\b',
        r'\bbless\s+your\s+(little\s+)?heart\b',
        r'\byou[\'re]+\s+so\s+cute\s+when\b',
        r'\baww+\s+(how\s+)?cute\b',
        r'\badorable\s+that\s+you\s+think\b',
        r'\brun\s+along\b',
        r'\blet\s+the\s+adults\b',
    ],
    "3e_demonstration_of_power": [
        r'\bstay\s+clear\s+of\s+you\b',
        r'\byour\s+poison\b',
        r'\bhandlers\b',
        r'\byour\s+script\b',

        r'\bown\s+you\b',
        r'\byou\s+work\s+for\s+(me|us)\b',
        r'\bi\s+made\s+you\b',
        r'\bwithout\s+(me|us)\s+you[\'re]+\s+nothing\b',
        r'\byou\s+answer\s+to\b',
        r'\bput\s+you\s+in\s+your\s+place\b',
        r'\bsilence\s+(her|you)\b',
    ],
    "3g_appearance_attack": [
        r'\bugly+\b',
        r'\bfea\b',
        r'\bhorse\s*teeth?\b',
        r'\blipstick\s+on\s+(a\s+)?pig\b',
        r'\balley\s+cat\b',
        r'\bdonkey+s?\b',
        r'\blook\s+pretty\b',
        r'\bpay\s+to\s+look\b',
        r'\bwhich\s+animal\s+are\s+you\b',
        # NEW
        r'\bdog\s*face\b',
        r'\bmanface\b',
        r'\bman\s+jaw\b',
        r'\bbutterface\b',
        r'\bneeds?\s+(a\s+)?makeover\b',
        r'\bwear\s+(a\s+)?bag\b',
        r'\bunch\b',
        r'\btroll\b',
        r'\bbag\s+over\s+(her|your)\s+head\b',
        r'\bno\s+one\s+would\s+(want|date|touch)\b',
        r'\bunfuckable\b',
        r'\bwould\s+not\s+touch\s+with\b',
    ]
}

# Compile all patterns (normalization now handles leet, so no [o0] needed in patterns)
compiled_patterns = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in keyword_patterns.items()
}



gender_markers = [
    r'\bgirls?\b', r'\bwomen\b', r'\bfemales?\b',
    r'\bladies\b', r'\blady\b', r'\bshe\b', r'\bher\b',
    r'\baoc\b', r'\balexandria\b', r'\bmegan\b',
]

degrading_words = [
    r'\bdumb\b', r'\bstupid\b', r'\bidiot\b', r'\bmoron\b',
    r'\bretard\b', r'\bclueless\b', r'\bincompetent\b',
    r'\bairhead\b', r'\bbrainless\b', r'\bdimwit\b',
    r'\bimbecile\b', r'\bignorant\b', r'\bdumbass\b',
    r'\bbuffoon\b', r'\bfool\b', r'\bdummy\b',
    r'\bnaive\b', r'\bdelusional\b', r'\bcrazy\b',
    r'\bhysterical\b', r'\bemotional\b', r'\birational\b',
]

dismissal_words = [
    r'\bshut\s+up\b', r'\bno\s+one\s+cares?\b', r'\bnobody\s+asked\b',
    r'\bgo\s+away\b', r'\bstay\s+in\s+your\s+lane\b',
    r'\bstick\s+to\b', r'\bsit\s+down\b', r'\bquiet\b',
    r'\bstop\s+talking\b', r'\bnot\s+your\s+business\b',
]

compiled_gender_markers = [re.compile(p, re.IGNORECASE) for p in gender_markers]
compiled_degrading_words = [re.compile(p, re.IGNORECASE) for p in degrading_words]
compiled_dismissal_words = [re.compile(p, re.IGNORECASE) for p in dismissal_words]


def check_gender_degrading_combo(reply_text):
    """Check for gender marker + degrading/dismissal word within ±100 chars"""
    if not isinstance(reply_text, str):
        return False, None

    normalized = normalize_text(reply_text)

    for gender_pattern in compiled_gender_markers:
        for gender_match in gender_pattern.finditer(normalized):
            gender_pos = gender_match.start()
            gender_word = gender_match.group()

            window_start = max(0, gender_pos - 100)
            window_end = min(len(normalized), gender_pos + 100)
            window = normalized[window_start:window_end]

            # Gender + degrading
            for deg_pattern in compiled_degrading_words:
                if deg_pattern.search(window):
                    deg_word = deg_pattern.search(window).group()
                    return True, f"2c_gender_combo: {gender_word} + {deg_word}"

            # Gender + dismissal
            for dis_pattern in compiled_dismissal_words:
                if dis_pattern.search(window):
                    dis_word = dis_pattern.search(window).group()
                    return True, f"3e_dismissal_combo: {gender_word} + {dis_word}"

    return False, None


def classify_reply(reply_text):
    if not isinstance(reply_text, str):
        return "non-misogynistic", None

    normalized = normalize_text(reply_text)

    for category, patterns in compiled_patterns.items():
        for pattern in patterns:
            match = pattern.search(normalized)
            if match:
                return "misogynistic", f"{category} (matched: '{match.group()}')"

    is_combo, combo_category = check_gender_degrading_combo(reply_text)
    if is_combo:
        return "misogynistic", combo_category

    return "non-misogynistic", None


#classification

df["classifier_label"], df["matched_category"] = zip(
    *df["Reply"].apply(classify_reply)
)



def is_misogynistic_label(label):
    if pd.isna(label):
        return 0
    parts = str(label).replace(';', ',').split(',')
    if any(p.strip().startswith('2') for p in parts):
        return 2
    if any(p.strip().startswith('3') for p in parts):
        return 3
    return 0

df['actual_binary'] = df['Code'].apply(is_misogynistic_label)
df['predicted_binary'] = (df['classifier_label'] == 'misogynistic').astype(int)


tp = ((df['actual_binary'] != 0) & (df['predicted_binary'] == 1)).sum()
fn = ((df['actual_binary'] != 0) & (df['predicted_binary'] == 0)).sum()
fp = ((df['actual_binary'] == 0) & (df['predicted_binary'] == 1)).sum()
tn = ((df['actual_binary'] == 0) & (df['predicted_binary'] == 0)).sum()



total = len(df)
actual_misogyny = (df['actual_binary'] != 0).sum()
percent_misogyny = (actual_misogyny / total) * 100 if total > 0 else 0

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
percent_fp_total    = (fp / total) * 100 if total > 0 else 0
percent_fn_total    = (fn / total) * 100 if total > 0 else 0



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
print(f"False Positive Rate (FPR):    {false_positive_rate:.2f}")
print(f"False Negative Rate (FNR):    {false_negative_rate:.2f}")
print(f"False Positives (% of total): {percent_fp_total:.2f}%")
print(f"False Negatives (% of total): {percent_fn_total:.2f}%")



explicit_actual = df['Code'].apply(
    lambda x: any(p.strip().startswith('2') for p in str(x).replace(';', ',').split(','))
)
implicit_actual = df['Code'].apply(
    lambda x: any(p.strip().startswith('3') for p in str(x).replace(';', ',').split(','))
    and not any(p.strip().startswith('2') for p in str(x).replace(';', ',').split(','))
)

total_explicit = explicit_actual.sum()
total_implicit = implicit_actual.sum()

explicit_caught = (explicit_actual & (df['predicted_binary'] == 1)).sum()
implicit_caught = (implicit_actual & (df['predicted_binary'] == 1)).sum()

explicit_catch_rate = (explicit_caught / total_explicit * 100) if total_explicit > 0 else 0
implicit_catch_rate = (implicit_caught / total_implicit * 100) if total_implicit > 0 else 0

print(f"\n--- MISOGYNY TYPE BREAKDOWN ---")
print(f"Explicit (code 2): {total_explicit} ({(total_explicit/total)*100:.2f}%)")
print(f"Implicit (code 3): {total_implicit} ({(total_implicit/total)*100:.2f}%)")

print("\n--- DETECTION BY MISOGYNY TYPE ---")
print(f"Explicit misogyny (code 2):")
print(f"  Total:   {total_explicit}")
print(f"  Caught:  {explicit_caught} ({explicit_catch_rate:.2f}%)")
print(f"  Missed:  {total_explicit - explicit_caught} ({100 - explicit_catch_rate:.2f}%)")
print(f"\nImplicit misogyny (code 3):")
print(f"  Total:   {total_implicit}")
print(f"  Caught:  {implicit_caught} ({implicit_catch_rate:.2f}%)")
print(f"  Missed:  {total_implicit - implicit_caught} ({100 - implicit_catch_rate:.2f}%)")
print(f"\nExplicit detection rate: {explicit_catch_rate:.2f}%")
print(f"Implicit detection rate: {implicit_catch_rate:.2f}%")
print(f"Difference:              {explicit_catch_rate - implicit_catch_rate:.2f} pp")


print("\n--- MATCHED CATEGORY BREAKDOWN (true positives only) ---")
tp_df = df[(df['actual_binary'] != 0) & (df['predicted_binary'] == 1)]
category_counts = (
    tp_df['matched_category']
    .str.extract(r'^(\w+)')[0]
    .value_counts()
)
for cat, count in category_counts.items():
    print(f"  {cat}: {count}")



false_positives_df = df[(df['actual_binary'] == 0) & (df['predicted_binary'] == 1)]
false_negatives_df = df[(df['actual_binary'] != 0) & (df['predicted_binary'] == 0)]
true_positives_df  = df[(df['actual_binary'] != 0) & (df['predicted_binary'] == 1)]
true_negatives_df  = df[(df['actual_binary'] == 0) & (df['predicted_binary'] == 0)]

# print("\n--- FALSE NEGATIVES (missed misogyny) ---")
# for _, row in false_negatives_df.iterrows():
#     print(f"\nReply: {row['Reply']}")
#     print(f"Actual label: {row['Code']}")
#     print(f"Why misogynistic: {row.get('Justification', 'N/A')}")

# print("\n--- FALSE POSITIVES (wrongly flagged) ---")
# for _, row in false_positives_df.iterrows():
#     print(f"\nReply: {row['Reply']}")
#     print(f"Actual label: {row['Code']}")
#     print(f"Matched category: {row['matched_category']}")

# print(f"\n--- TRUE POSITIVES ({len(true_positives_df)}) ---")
# for _, row in true_positives_df.iterrows():
#     print(f"\nReply: {row['Reply']}")
#     print(f"Matched category: {row['matched_category']}")
#     print(f"Actual label: {row['Code']}")

false_negatives_df.to_csv("false_negatives.csv", index=False)
false_positives_df.to_csv("false_positives.csv", index=False)
true_positives_df.to_csv("true_positives.csv", index=False)



#more evaluation

# Load the full annotated dataset (all rows with ground truth)
misogynistic_df = df[df['actual_binary'] != 0].copy()
non_misogynistic_df = df[df['actual_binary'] == 0].copy()

# Finds words that appear disproportionately in misogynistic replies

stopwords = {
    'the','a','an','is','it','to','of','and','in','you','i','that','this',
    'her','she','he','they','we','are','was','be','have','has','for','on',
    'not','do','but','with','at','your','my','so','just','get','can','will',
    'its','im','dont','youre','they','all','about','what','if','or','no',
    'me','up','out','go','like','know','think','really','rt','amp','https'
}

def get_word_freq(texts):
    words = []
    for text in texts.dropna():
        normalized = normalize_text(str(text))
        words.extend(re.findall(r'\b[a-z]{3,}\b', normalized))
    return Counter(w for w in words if w not in stopwords)

misog_freq   = get_word_freq(misogynistic_df['Reply'])
nonmisog_freq = get_word_freq(non_misogynistic_df['Reply'])

total_misog    = sum(misog_freq.values())
total_nonmisog = sum(nonmisog_freq.values())

# Score = how much more common a word is in misogynistic vs non-misogynistic replies
print("\n─── TOP UNIGRAMS IN MISOGYNISTIC REPLIES (vs non-misogynistic) ───")
print(f"{'Word':<20} {'Misog%':>8} {'NonMisog%':>10} {'Ratio':>8} {'Count':>7}")
print("─" * 60)

word_scores = []
for word, count in misog_freq.most_common(500):
    if count < 3:
        continue
    misog_pct    = count / total_misog
    nonmisog_pct = (nonmisog_freq.get(word, 0) + 1) / total_nonmisog  # +1 smoothing
    ratio = misog_pct / nonmisog_pct
    word_scores.append((word, count, misog_pct, nonmisog_pct, ratio))

word_scores.sort(key=lambda x: x[4], reverse=True)
for word, count, mp, nmp, ratio in word_scores[:40]:
    print(f"{word:<20} {mp*100:>7.2f}% {nmp*100:>9.2f}% {ratio:>8.1f}x {count:>6}")


# Finds two-word phrases distinctive to misogynistic replies

def get_bigrams(texts):
    bigrams = []
    for text in texts.dropna():
        normalized = normalize_text(str(text))
        words = [w for w in re.findall(r'\b[a-z]{2,}\b', normalized) if w not in stopwords]
        bigrams.extend(zip(words, words[1:]))
    return Counter(bigrams)

misog_bigrams    = get_bigrams(misogynistic_df['Reply'])
nonmisog_bigrams = get_bigrams(non_misogynistic_df['Reply'])

total_misog_bi    = sum(misog_bigrams.values()) or 1
total_nonmisog_bi = sum(nonmisog_bigrams.values()) or 1

print("\n─── TOP BIGRAMS IN MISOGYNISTIC REPLIES ───")
print(f"{'Bigram':<30} {'Misog%':>8} {'NonMisog%':>10} {'Ratio':>8} {'Count':>7}")
print("─" * 70)

bigram_scores = []
for bigram, count in misog_bigrams.most_common(1000):
    if count < 3:
        continue
    mp   = count / total_misog_bi
    nmp  = (nonmisog_bigrams.get(bigram, 0) + 1) / total_nonmisog_bi
    ratio = mp / nmp
    bigram_scores.append((bigram, count, mp, nmp, ratio))

bigram_scores.sort(key=lambda x: x[4], reverse=True)
for bigram, count, mp, nmp, ratio in bigram_scores[:30]:
    phrase = ' '.join(bigram)
    print(f"{phrase:<30} {mp*100:>7.2f}% {nmp*100:>9.2f}% {ratio:>8.1f}x {count:>6}")


# Specifically looks at what your classifier is MISSING

fn_df = df[(df['actual_binary'] != 0) & (df['predicted_binary'] == 0)].copy()

print(f"\n─── PATTERN MINING: FALSE NEGATIVES (n={len(fn_df)}) ───")
print("These are misogynistic replies your classifier currently misses\n")

fn_freq    = get_word_freq(fn_df['Reply'])
fn_bigrams = get_bigrams(fn_df['Reply'])
total_fn   = sum(fn_freq.values()) or 1
total_fn_bi = sum(fn_bigrams.values()) or 1

print("Top distinctive UNIGRAMS in missed replies:")
print(f"{'Word':<20} {'FN%':>7} {'NonMisog%':>10} {'Ratio':>8} {'Count':>6}")
print("─" * 55)

fn_word_scores = []
for word, count in fn_freq.most_common(300):
    if count < 2:
        continue
    fp_pct   = count / total_fn
    nmp      = (nonmisog_freq.get(word, 0) + 1) / total_nonmisog
    ratio    = fp_pct / nmp
    fn_word_scores.append((word, count, fp_pct, nmp, ratio))

fn_word_scores.sort(key=lambda x: x[4], reverse=True)
for word, count, fp_pct, nmp, ratio in fn_word_scores[:25]:
    print(f"{word:<20} {fp_pct*100:>6.2f}% {nmp*100:>9.2f}% {ratio:>8.1f}x {count:>5}")

print("\nTop distinctive BIGRAMS in missed replies:")
print(f"{'Bigram':<30} {'FN%':>7} {'NonMisog%':>10} {'Ratio':>8} {'Count':>6}")
print("─" * 65)

fn_bigram_scores = []
for bigram, count in fn_bigrams.most_common(500):
    if count < 2:
        continue
    fp_pct = count / total_fn_bi
    nmp    = (nonmisog_bigrams.get(bigram, 0) + 1) / total_nonmisog_bi
    ratio  = fp_pct / nmp
    fn_bigram_scores.append((bigram, count, fp_pct, nmp, ratio))

fn_bigram_scores.sort(key=lambda x: x[4], reverse=True)
for bigram, count, fp_pct, nmp, ratio in fn_bigram_scores[:20]:
    phrase = ' '.join(bigram)
    print(f"{phrase:<30} {fp_pct*100:>6.2f}% {nmp*100:>9.2f}% {ratio:>8.1f}x {count:>5}")


# Compares what language looks like in code 2 vs code 3

explicit_df = df[explicit_actual].copy()
implicit_df = df[implicit_actual].copy()

explicit_freq = get_word_freq(explicit_df['Reply'])
implicit_freq = get_word_freq(implicit_df['Reply'])

total_exp = sum(explicit_freq.values()) or 1
total_imp = sum(implicit_freq.values()) or 1

print("\n─── WORDS MORE COMMON IN EXPLICIT (code 2) vs IMPLICIT (code 3) ───")
print(f"{'Word':<20} {'Explicit%':>10} {'Implicit%':>10} {'Ratio':>8}")
print("─" * 55)

exp_vs_imp = []
for word in set(explicit_freq) | set(implicit_freq):
    exp_pct = (explicit_freq.get(word, 0) + 1) / total_exp
    imp_pct = (implicit_freq.get(word, 0) + 1) / total_imp
    if explicit_freq.get(word, 0) >= 3:
        exp_vs_imp.append((word, exp_pct, imp_pct, exp_pct / imp_pct))

exp_vs_imp.sort(key=lambda x: x[3], reverse=True)
print("  >> More explicit:")
for word, ep, ip, ratio in exp_vs_imp[:15]:
    print(f"  {word:<20} {ep*100:>9.2f}% {ip*100:>9.2f}% {ratio:>8.1f}x")

print("  >> More implicit:")
for word, ep, ip, ratio in sorted(exp_vs_imp, key=lambda x: x[3])[:15]:
    print(f"  {word:<20} {ep*100:>9.2f}% {ip*100:>9.2f}% {ratio:>8.1f}x")


#export 
discovery_rows = []
for word, count, mp, nmp, ratio in word_scores[:100]:
    discovery_rows.append({
        'type': 'unigram', 'term': word, 'count_misogynistic': count,
        'pct_misogynistic': round(mp*100, 3),
        'pct_non_misogynistic': round(nmp*100, 3),
        'ratio': round(ratio, 2), 'source': 'all_misogynistic'
    })
for bigram, count, mp, nmp, ratio in bigram_scores[:100]:
    discovery_rows.append({
        'type': 'bigram', 'term': ' '.join(bigram), 'count_misogynistic': count,
        'pct_misogynistic': round(mp*100, 3),
        'pct_non_misogynistic': round(nmp*100, 3),
        'ratio': round(ratio, 2), 'source': 'all_misogynistic'
    })
for word, count, fp_pct, nmp, ratio in fn_word_scores[:50]:
    discovery_rows.append({
        'type': 'unigram', 'term': word, 'count_misogynistic': count,
        'pct_misogynistic': round(fp_pct*100, 3),
        'pct_non_misogynistic': round(nmp*100, 3),
        'ratio': round(ratio, 2), 'source': 'false_negatives'
    })

discovery_df = pd.DataFrame(discovery_rows)
discovery_df.to_csv("pattern_discovery.csv", index=False)
print("\n✓ Saved pattern_discovery.csv")