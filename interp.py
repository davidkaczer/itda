"""
Methods for running interpretation studies on the SAEs, both with human trials
and with automated methods.
"""

import argparse
import os
import random
import re

import numpy as np
import openai
import torch
from dotenv import load_dotenv
from ito import GPT2, OMP, OMP_L0, SEQ_LEN, load_model, OMPSAE
from tqdm import tqdm


def highlight_string(tokenizer, tokens, target_idx, window=16, highlight=True):
    str = ""
    for i, token in enumerate(tokens):
        # Skip tokens that aren't within `window` places of the target token
        if abs(i - target_idx) > window:
            continue

        token_str = tokenizer.decode(token).replace("\n", "")
        if highlight and (i == target_idx):
            str += f"<<<{token_str}>>>"
        else:
            str += f"{token_str}"
    return str


def get_strings_activations(acts, tokens, n=20):
    non_zero_indices = np.argwhere(acts != 0)
    zero_indices = np.argwhere(acts == 0)

    num_non_zero = int(n * 0.8)
    num_zero = n - num_non_zero

    try:
        non_zero_idxs = non_zero_indices[
            np.random.choice(
                non_zero_indices.shape[0], size=num_non_zero, replace=False
            )
        ]
        zero_idxs = zero_indices[
            np.random.choice(zero_indices.shape[0], size=num_zero, replace=False)
        ]

        idxs = np.concatenate((non_zero_idxs, zero_idxs))
        np.random.shuffle(idxs)
    except ValueError as e:
        raise ValueError(
            f"Not enough activations found to sample {n} values. Found {non_zero_indices.shape[0]} activations.",
            e,
        )

    sample_strs, sample_acts = [], []
    for inp, tok in idxs[:-1]:
        act = acts[inp, tok]
        str = highlight_string(tokens[inp], tok)
        sample_strs.append(str)
        sample_acts.append(act)

    sample_strs, sample_acts = zip(
        *sorted(zip(sample_strs, sample_acts), key=lambda x: x[1], reverse=True)
    )

    return sample_strs, sample_acts


def construct_comparison_question(strings, activations, answer_idxs):
    """
    Given a a list of token-tagged inputs and their activations, constructs a
    comparison question. I.e. given two strings, which has the higher
    activation?
    """
    prompt = ""
    for i, (a, s) in enumerate(zip(activations, strings)):
        if i in answer_idxs:
            continue
        prompt += f"{a:.2f} {s}\n"
    prompt += "\n\n"
    prompt += f"1) {strings[answer_idxs[0]]}\n"
    prompt += f"2) {strings[answer_idxs[1]]}\n"

    answer = 1 if activations[answer_idxs[0]] > activations[answer_idxs[1]] else 2

    return prompt, answer


def predict_comparison_answers(prompts, answers):
    """
    Runs auto-interp on comparison questions and returns the accuracy.
    """
    predictions = []
    for prompt in tqdm(prompts, "Running auto-interp on comparison questions..."):
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are doing autointerp on SAE features. First you will be provided with a list of activations and strings. Then you will be asked to compare two strings based on their activations. Answer only with the number of the string you think has the higher activation.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        predictions.append(int(answer))
    correct = sum([1 for p, a in zip(predictions, answers) if p == a]) / len(answers)
    return correct


def get_nonzero_activations(sparse_activations, count=20, get_zero=False):
    if get_zero:
        # Get sequences where the feature is not active
        idxs = np.nonzero((sparse_activations > 0).sum(axis=1) == 0)
        idxs = np.column_stack(
            [idxs, np.random.choice(sparse_activations.shape[1], idxs.shape[0])]
        )
    else:
        idxs = np.nonzero(sparse_activations > 0)

    try:
        sample = np.random.choice(idxs.shape[0], count, replace=False)
    except:
        raise ValueError(
            f"Not enough {'zero' if get_zero else 'non-zero'} activations found to sample {count} values. Found {idxs.shape[0]} activations."
        )
    idxs = idxs[sample]
    acts = sparse_activations[idxs[:, 0], idxs[:, 1]]
    if get_zero:
        acts = np.zeros_like(acts)
    return idxs, acts


def generate_latent_description(
    sparse_activations,
    tokenizer,
    token_dataset,
    latent,
    active_example_count=20,
    inactive_example_count=10,
):
    nonzero_idxs, nonzero_activations = get_nonzero_activations(
        sparse_activations[:, :, latent], active_example_count
    )
    zero_idxs, zero_activations = get_nonzero_activations(
        sparse_activations[:, :, latent],
        inactive_example_count,
        get_zero=True,
    )

    # Sort the indices by activations
    all_indices = np.concatenate((nonzero_idxs, zero_idxs), axis=0)
    all_activations = np.concatenate((nonzero_activations, zero_activations))
    all_indices, all_activations = zip(
        *sorted(zip(all_indices, all_activations), key=lambda x: x[1], reverse=True)
    )
    prompt = ""
    for (seq_idx, token_idx), act in zip(all_indices, all_activations):
        s = highlight_string(tokenizer, token_dataset[seq_idx], token_idx)
        prompt += f"{float(act):.2f} {s}\n"

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are tasked with describing the behaviour of SAE features in such a way that your description can be used to predict the activation on inputs. Try to explain examples across the entire distribution, not just the first few. For example:

Prompt:
0.65  (Additional file [1](#S1){ref-type="supplement<<<ary>>>-material"}). In the absence of a karyotype, our assessment of
0.44  pad areas and the protective layer regions covered by the opaque strip conductor areas of the<<< secondary>>> film remain. The base plate developed in this way is etched, whereby the laid
0.33  by a devastating knee injury – another piece of evidence some will stretch to use as<<< support>>> for the curse. A much more reasonable explanation exists. Part of what made D
0.33 ol and melanin biosynthetic pathways. Supporting to this hypothesis, production of<<< secondary>>> metabolites by *A*. *flavus*, *A*. *paras
0.30 *elements enriched in the promoters of tomato MTases genes.Click here for<<< additional>>> data file.
0.26  an excitation power of −30��dBm and, except for stronger<<< background>>> noise, we found identical spin-wave characteristics. We used a frequency sweep method
0.18 idylserine externalization, all hallmark characteristics of apoptosis. These effect<<<or>>> caspases then cleave PARP ([@B12]), and the activated
0.11 <|endoftext|> from a mixture comprising the two types of polypept<<<ide>>> dimers is reported in US 2005/0163782. Bispecific tet
0.08 <|endoftext|>: str            The<<< component>>> of the seismic moment tensor to plot. ``"full"`` (the
0.00 <|endoftext|>] [<<< 1>>>  4  9  3  4  3  3  2  9
0.00  would be his second floor bedroom and his own back yard."It<<< would>>> be nice," he said. "You
0.00 4 –.6 –.7 –.7 –.9, 6+<<<),>>> but ��Que Buena 93.3�� slips three-tenth
0.00 aI restriction site in the PARK2 gene (Figure  [6](#<<<F>>>6){ref-type="fig"}). This gene encodes an E3
0.00 1843_1_En_8_Chapter_IEq5.gif<<<)>>> beyond its extremities. To extend such a path beyond S, attributes not present
0.00  wouldn't work. I need to send the IP back to the user (my<<< mobile>>> phone) so they can link up with the proper server. Loopback wouldn't
0.00  oral09262011-1-28     <<< >>>                
0.00 iously, perhaps wondering if they might expire while doing the exam, which could reflect<<< badly>>> on the college. "Well, if you're sure...""I
0.00  decreased, his weight increased, and head control was achieved after initiation of hydrochlor<<<oth>>>iazide.Assessment of the patient's family history revealed that his
0.00 **4/12 + z**2 + 103*z - 3. Give<<< 5>>>*s(k) - 4*v(k).-k**
-0.11 But you can't kill her?""Not yet.""<<<Why>>> not?""I've explained that a hundred times too. No attack
-0.11  ([@b83]--[@b85]). Emerging clinical evidence suggests that<<< dietary>>> modification to increase polyphenol intakes from whole-food sources can lead to improved
-0.12 D. in political science with a focus on international political economy, and is professor<<< at>>> the Balsillie School of International Affairs and the University of Waterloo. She
-0.13  300; 77.0% were uninsured and dependent on the public sector. The<<< large>>> uninsured proportion of the population skews the quadruple burden of disease towards the public
-0.14  day was much stronger than our modern engagements are. One had to get a divorce<<< to>>> break it -- it was unconsummated marriage.)Joseph full
-0.14  the material is to be deposited. It is further desirable to provide a method for<<< electro>>>less deposition that provides a high conductivity layer of deposited material that adheres well
-0.15 centration of *A*. *terreus* Acetyl-Co<<<A>>> in response to subculturing and amendment with various concentrations of *P*. *
-0.19  good news is that merchants are responding to shifting consumer behaviors and implementing key features that<<< smooth>>> the path to purchase, especially across touchpoints. We��ve long recommended
-0.22  {        // Key must be UnsafeAllowNoneSignatureType to prevent<<< accidentally>>>       // accepting 'none' signing method      if _, ok :=
-0.23  making "after eight" chocolateI am following a very simple recipe for<<< making>>> peppermint-stuffed chocolate, like the "After Eight" kind
-0.23  filing financial reports with the government.Foreign organizations like the MacArthur Foundation have<<< shut>>>tered their offices in Russia in recent years in response to the foreign agent law.
Description: Activates on tokens relating to secondary or supplementary information of material.

Prompt:
0.89 Suppose 0 = -3*q + 39 + 18. Suppose 4*<<<w>>> - 7 = -q. Let z(s) be the second derivative of
0.83  0.031395.00 ± 26.09Abrod<<<w>>>um9.75 ± 0.0130.00 ± 
0.73  "I didn't give you that apple." "You took it." "O<<<w>>>!" "Hey!" "Hold on, you hairy, little thief." "Come
0.73  + stop_words_len + topic1_len + i]:copy(<<<w>>>2vutils:phrase
0.67 This model was first considered in [@Gauntlett:2009z<<<w>>>]. It contains just one massive vector multiplet and one hypermultiplet,
0.53 **3) to the form d + m*w**4 + l*<<<w>>> + z*w**2 + s*w**3 and give s.
0.49  the form d + m*w**4 + l*w + z*<<<w>>>**2 + s*w**3 and give s.-
0.16  very limited in the back of the electronic device and there is inadequate room for a<<< wrench>>>. For example, the cable box or television may be located within an entertainment console
0.13  which the Indian Government has recently unveiled, and the positive effects should be visible sooner<<< rather>>> than later.��He said the implementation of public-private partnership
0.11  available control units offer xe2x80x9cextended functions.<<<xe>>>2x80x9dExtended functions implement features above and beyond basic device
0.09 \]. Oxidative stress, by definition, is a biochemical condition in which<<< oxid>>>ant species overwhelm antioxidant defense ultimately leading to a given biological damage \[[@ref
0.08  the United Kingdom, supports the Republic of Ireland leaving the European Union (an '<<<Ire>>>xit'), and is the President of the Irish Freedom Party, a party that advocates
0.07 <|endoftext|> the equivalent<<< of>>> windows mobile or the iphone OSGeneva is a minimalist personal or corporate
0.00  Exit��.There��s something quite cold, then,<<< about>>> the archaeologist or historian who treats this or that ten-year or even fifty
0.00 <|endoftext|> get a bigger appartment because<<< she>>> couldn't share the rent fifty-fifty." "And I also think she
0.00 <|endoftext|>0.<<<62>>>5051;-0.780584;,   0.000000
0.00   "dur_s2_1"    -200 <<< >>>   -49 next_cvox==-     
0.00  combined length when laying flat is much longer than the available length that the wall panels<<< provide>>> when they are in their folded flat configuration. An attempt to collapse the roof panels
0.00  alteration of a single base in the stack can either increase or decrease the conductivity<<< of>>> the dsDNA helix, depending on the type of the mismatched.
0.00         'Notification' => Illuminate\<<<Support>>>\
0.00  mother is sending me your birth certificate so that you can get a passport, so<<< we>>> can fly to Rome in a couple of weeks." Whaaa?? Out of the
0.00 <|endoftext|>:23 GMT+0000 (WET)</div><<<</>>>div><script src="../../../../prettify.js
0.00 ed by permission of the author.AHARON SHABTAI<<<:>>> "The Fence" by Aharon Shabtai, reprinted by permission
-0.09  your life. It's about knowing where your money is going, and that will<<< make>>> your life a whole lot easier."If you plan out exactly how to
-0.11  find me that night? Boston!��CCME Names Radha<<< Sub>>>
-0.13 AdS_2\times S^2$) or hyperbolic ($<<<Ad>>>S_2\times \HH^2$) horizons. The modifications
-0.14 0x3333333333333333);      Reg2 = _mm_s<<<lli>>>_epi32(Reg1, 2);   Reg1 = _
-0.14  important end and well to the Machinery Hall, where assigned hotel George H.<<< Cor>>>liss, theory mess of Rhode Island, needed viewing at the tribological shipping
-0.15  with the country��s other sovereign fund Mubadala.S<<<ar>>>awak Report has further exposed how KAQ used Aabar to finance
-0.17  "I mean, we know that we are going to break up eventually." "<<< Definitely>>>." " Are your parents
Description: Activates on the letter w.

Prompt:
0.66 She's calling from Hartford: another young dark-skinned man has been killed—<<<shot>>> by police in the head while lying on the ground
0.36  last week not to charge the police officer who fatally shot 18-year-old<<< Michael>>> Brown. Violent protests and looting erupted after the decision, resulting in at least a
0.30  Being something of a reckless hot-head herself, she appreciated the help of the<<< SWAT>>> Kats and would work openly with them. Feral himself faced something of a dilemma
0.29  fold our tents. Yet in fact, throughout the world, transfusions of poetic<<< language>>> can and do quite literally keep bodies and souls together—and more.
0.29 fundamental right�� necessary to preserve ��an individual right to self-<<<defense>>>.�� McDonald also reiterated Heller��s assertion that ��self-
0.24 ified force."I guess the only difference is when football players use<<< excessive>>> force, they get penalized," Stewart said.Stewart then showed
0.22  his wallet. The victim broke away and ran toward a nearby house and was shot<<< in>>> the back. He obtained aid and survived.Four hours after the robbery
0.20 Monday, November 30, 2009Black News: Towanna Freeman on Domestic<<< Violence>>>One in every four women
0.20  to that of Southern Taiwan. Precipitation is abundant throughout the year; the<<< rain>>>iest month is August while the driest month is April. November to February are
0.19 <|endoftext|>inspector Ravinderpal Singh was shot dead<<< from>>> point blank range in Chehertha locality in Amritsar following an argument
0.18  which has a 75-mph speed limit, one in four drivers going above 80<<< mph>>>. In California, where the speed limit is 70 mph, one in five drivers
0.17  experiencing a one-year wealth increase of 17 percent, they��ve had<<< money>>> to spend.The financial sector spent $400 million on lobbying in the
0.17  of evolutionary materialism, that matter, energy, space, time and resulting blind<<< chance>>> and mechanical necessity are all that exist, we do inescapably end up in
0.14  companies and the workers were carried on in San Francisco and Seattle, the dispute could<<< not>>> be said to be 'at' the Alaskan establishments as
0.13  hvordan flere af unge venstrefløjsaktiv<<<ister>>> de senere ��r har måttet flytte fra by
0.12 rius rose, howling with rage, and prepared to throw himself at Kalix<<<.>>> At that moment, there was a furious banging on the door."
0.10 ," he said.Smith also questioned Doherty about the description he gave<<< of>>> his assailant when interviewed in the emergency room at Catholic Medical Center. She said he
0.08  Googling and that makes the anxiety felt subside, soon enough more intrusive<<< thoughts>>> will pop up, doubt will return and the sufferer will feel like he/
0.07 , J.In this case, defendant was charged with reckless driving, which<<< is>>> a misdemeanor.[1] In his ensuing jury trial, defendant asked the trial court
0.00  Netanyahu handled this situation adeptly: by personally calling Erdoğan and thanking him for<<< sending>>> the planes, he got credit for decency in the eyes of the world, and
0.00  in [@ross] to study malaria and later on, basic compartmental models<<< to>>> study infectious diseases were established in a sequence of three papers by Kermack and
0.00  they get, should they get any at all.It��s<<< unlikely>>> that Spurs will approach this game in any other way than how they usually do –
0.00  the godwits�� migration strategy across the Pacific.Here in<<< New>>> Zealand, the greatest climate change threat to godwits comes from rising sea levels
0.00 ianas o con partidos confesionales. Hay que liberar<<< al>>> mundo del mito y de la mística, para dejar
0.00  - The 16th-ranked Oregon Ducks look to remain atthe top of<<< the>>> Pac-12 standings when the they host the upset-minded WashingtonHusk
0.00  true./ The voice sounded oddly smug – pleased, even. That left Bruce more<<< dis>>>concerted than it should have. /I see mine./Yes,
0.00 , the rules of both contribution and indemnity could apply where a seller does not<<< contribute>>> to a defect in a product, but commits an independent act of negligence or is
0.00 01-0118v.         <<< >>>                
0.00 ., if![$$\\vert \\mathcal{E}\\vert<<< =>>> n_{\\mathcal{E}}$$](A271843_
-0.18  inside!" "Do it now!" "Okay!" "Ooh!" "E<<<w>>>!" "Willie, we are going to die!" "Lt's soft."
Description: Activates on language relating to violence or conflict.
""",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
    match = re.search(r"Description:\s*(.*)", text)
    if match:
        return match.group(1)
    else:
        raise ValueError(
            f"No description found in response.\n\nResponse: {text}\n\nPrompt: {prompt}"
        )


def detection_eval(
    sae, model_activations, tokenizer, token_dataset, n_latents=100
):
    # OMP SAEs use a dense format for activations
    print("Getting sample of latents...")
    explanation_active_example_count, explanation_inactive_example_count = 20, 10
    if len(activations) == 2:
        # TODO: Get the activations and indices here rather than loading from disk
        # TODO: Probably easier to commonise the interface of OMP with regular SAEs.
        sae_indices, sae_activations = activations
        latents, counts = np.unique(sae_indices.flatten(), return_counts=True)
        # TODO: 0.8 * train_size is quite hacky to make sure we don't get
        # latents that are always active. Must be a better way.
        filtered_latents = latents[
            (explanation_active_example_count < counts) & (counts < (train_size * 0.8))
        ]
        sampled_latents = np.random.choice(filtered_latents, n_latents, replace=False)

        sparse_activations = np.zeros((*sae_activations.shape[:2], n_latents))
        for i, latent in enumerate(sampled_latents):
            mask = sae_indices == latent
            sparse_activations[:, :, i] = (sae_activations * mask).sum(axis=-1)
        # TODO: Fix the weird mix of torch and numpy
        sparse_activations = torch.from_numpy(sparse_activations)
    else:
        # Do this in two steps because the activations are too large to fit in
        # memory
        # 1) Filter out the latents that are active in too few or too many
        # examples, and randomly sample
        # 2) Get the activations for the sampled latents. We want latents that
        # have more than 20 activations and more than 10 samples where they are
        # inactive.

        sae_latent_frequencies = torch.zeros(sae.W_dec.size(0), device=sae.W_dec.device)
        sae_sample_frequencies = []
        for batch in tqdm(
            torch.split(model_activations, 32),
            desc="Getting SAE activations...",
        ):
            acts = sae.encode(batch.to(sae.W_dec.device))
            sae_latent_frequencies += (acts > 0).sum(dim=(0, 1))
            sae_sample_frequencies.append((acts == 0).all(dim=1))
        sae_sample_frequencies = torch.cat(sae_sample_frequencies, dim=0).sum(dim=0)
        
        filtered_latents = (
            sae_latent_frequencies > explanation_active_example_count
        ) & (sae_sample_frequencies > explanation_inactive_example_count)
        sampled_latents = np.random.choice(
            np.nonzero(filtered_latents.cpu().numpy())[0], n_latents, replace=False
        )

        activations = []
        for batch in tqdm(
            torch.split(model_activations, 32), desc="Getting SAE activations..."
        ):
            acts = sae.encode(batch.to(device))[:, :, sampled_latents]
            activations.append(acts.cpu())
        sparse_activations = torch.cat(activations, dim=0)

    correct = 0
    failed = 0
    for latent in tqdm(range(n_latents), desc="Running detection eval..."):
        try:
            latent_description = generate_latent_description(
                sparse_activations,
                tokenizer,
                token_dataset,
                latent,
                active_example_count=explanation_active_example_count,
                inactive_example_count=explanation_inactive_example_count,
            )
        except ValueError as e:
            print(f"Failed to generate description for latent {latent}.", e)
            failed += 1
            continue

        nonzero_count = 5
        nonzero_idxs, nonzero_acts = get_nonzero_activations(
            sparse_activations[:, :, latent], count=nonzero_count
        )

        zero_count = 20
        zero_idxs, _ = get_nonzero_activations(
            sparse_activations[:, :, latent], count=zero_count, get_zero=True
        )

        answer = random.randint(0, 4)
        try:
            options_sample = np.random.choice(zero_idxs.shape[0], size=5)
            options = zero_idxs[options_sample]
        except TypeError as e:
            print("Failed to sample zero activations:", e, zero_idxs)
            failed += 1
            continue
        options[answer] = random.choice(nonzero_idxs)

        prompt = f"""Description: {latent_description}

Which of the following inputs activates on this feature?
"""
        for i, idx in enumerate(options):
            s = highlight_string(
                tokenizer, token_dataset[idx[0]], idx[1], highlight=False
            )
            prompt += f"{i}) {s}\n"

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You will be provided with a description of a feature and a list of inputs. One of those inputs activates the feature. Select the input that activates the feature, reply only with the number of the input and no further detail.

These are examples, but don't include 'Answer:' in your response.                    

Description: Activates on tokens related to sections, parts, or pieces of material or information.

Which of the following inputs activates on this feature?
0) ol/l ascorbate to normalise the results according to respective mitochondrial content in wild-type and liver-specific *Ampkα1/2
1)  all telecommunications carriers and certain non-telecommunications carriers to contribute in proportion to their share of end-user telecommunications revenues. See Order ¶¶ 39-42. The
2)  when the engine is operating under a heavy load at a high speed, a part of the air introduced into the inlet passage portion is fed into the helical portion
3)  environment around Luoyang bridge officially submitted to national culture relics bureau, and most parts of the plan got approval in December, expect the part for dredging.
4)  in May 1902. Later, similar work had to continue in order to strengthen the rest of the columns of the Temple.See alsoTalatat

Answer: 3

Description: Activates on tokens related to games, sessions, or competitive events.

Which of the following inputs activates on this feature?
0)  thanks to tiebreakers with Robb but since Daniel gave Robb 2 points on the last turn in order to spite Pounder from getting 1 food I'm going to take a
1)  good chance the Czech forward could push his point streak to four games in tonight's contest.NHL and the NHL Shield are registered trademarks and
2)         // Allocate session data for the newly established session            LoggingSession*
3)  bred," the queen had said with displeasure.It had quieted the council, but the problem wasn't going away. For some reason, the entire H
4) <|endoftext|>' is for the game Nethack, but that's not quick unless the player is unlucky or careless

Answer: 4

Description: Activates on tokens related to sending, providing, or distributing information or services.

Which of the following inputs activates on this feature?
0)  fire. He initiated fire."Doherty said Webster came closer as he fired, So close Doherty could see the muzzle blast."I thought
1)  comfortable, though that was something she would never have admitted. Daniel meanwhile was distributing pizza. Next
2) <|endoftext|>ker with Chelsea also appeared and provided corroborative testimony.[2]  In stating that both Chelsea and Far
3)  written in two stages. In the first stage of composition, circa 1230, Guillaume de Lorris wrote 4,058 lines describing a courtier's
4) m guilty of it myself. We don��t grasp the reigns and start a new path, yet we try to reach new, undiscovered lands by following

Answer: 1
""",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        response = response.choices[0].message.content.strip()
        try:
            response = int(response)
            if int(response) == answer:
                correct += 1
        except ValueError:
            print("Invalid response:", response)
            failed += 1

    return correct / (n_latents - failed)


if __name__ == "__main__":
    load_dotenv(override=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=GPT2)
    parser.add_argument("--l0", type=int, default=OMP_L0)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--ito_fn", type=str, default=OMP)
    parser.parse_args()
    args = parser.parse_args()

    # TODO: Just reconstruct the activations each time rather than bothering
    # saving them?
    model_activations = torch.load(f"data/{args.model}/model_activations.pt")

    model, saes, token_dataset = load_model(args.model, device=device)
    sae = saes[0]

    atoms = torch.load(f"data/{args.model}/atoms.pt")
    ito_sae = OMPSAE(model, atoms, device=device)

    # assume the test set is at the end of the dataset
    test_size = int(model_activations.shape[0] * 0.3)
    train_size = len(token_dataset["tokens"]) - test_size
    token_dataset = token_dataset["tokens"][train_size:, : args.seq_len]

    feature_count = 100
    sample_string_count = 20

    print("Running ITO detection eval...")
    ito_detection_score = detection_eval(
        ito_sae,
        model_activations[train_size:, : args.seq_len],
        model.tokenizer,
        token_dataset,
        n_latents=feature_count,
    )

    print("Running SAE detection eval...")
    sae_detection_score = detection_eval(
        sae,
        model_activations[train_size:, : args.seq_len],
        model.tokenizer,
        token_dataset,
        n_latents=feature_count,
    )
    print(f"ITO detection score: {ito_detection_score}")
    print(f"SAE detection score: {sae_detection_score}")