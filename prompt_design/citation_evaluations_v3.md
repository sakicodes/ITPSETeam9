## 1. Hireability
Higgins, C. A., & Judge, T. A. (2004). The effect of applicant influence tactics on recruiter perceptions of fit and hiring recommendations: A field study. *Journal of Applied Psychology, 89*(4), 622–632.

> "Recruiters were also asked about their hiring recommendations for each applicant. Specifically, recruiters were asked the likelihood that they would recommend hiring the applicant ('I would recommend extending a job offer to this applicant') and for their overall evaluation of the applicant ('Overall, I would evaluate this applicant positively'). Recruiters responded using the same 7-point scale (1 = strongly disagree to 7 = strongly agree)... These two items were combined to form a hiring recommendation variable similar to that used by Cable and Judge (1997). In the present study, the internal consistency reliability was α = .92."

---

## 2. Perceived Competence

**Fiske, S. T., Cuddy, A. J. C., Glick, P., & Xu, J. (2002). A model of (often mixed) stereotype content: Competence and warmth respectively follow from perceived status and competition. *Journal of Personality and Social Psychology, 82*(6), 878–902.**

Item wording verified via a later review by one of the original authors: Fiske, S. T. (2018). Stereotype content: Warmth and competence endure. *Current Directions in Psychological Science, 27*(2), 67–73.

> "Competence items include competent, intelligent, skilled, and efficient, as well as assertive and confident."


---

## 3. Person–Job Fit
Higgins & Judge (2004), same paper as above

> "We assessed perceived P–J fit using two statements designed to determine the congruence between the demands of the job and the abilities of the applicant. The first statement was 'This applicant possesses the KSAs necessary to perform the duties of this specific job.' The second statement was 'I believe this applicant can achieve a high level of performance in this particular job.' The coefficient alpha reliability of this scale in the present study was α = .89."


---

## 4. Expected Job Performance

**Williams, L. J., & Anderson, S. E. (1991). Job satisfaction and organizational commitment as predictors of organizational citizenship and in-role behaviors. *Journal of Management, 17*(3), 601–617.**

In-Role Behavior (IRB) subscale, 7 items (Table 1, p. 606):

> 1. Adequately completes assigned duties.

> 2. Fulfills responsibilities specified in job description.

> 3. Performs tasks that are expected of him/her.

> 4. Meets formal performance requirements of the job.

> 5. Engages in activities that will directly affect his/her performance evaluation.

> 6. Neglects aspects of the job he/she is obligated to perform.

> 7. Fails to perform essential duties. 


---

## 5. Leadership Potential

**J. S. Mueller, J. A. Goncalo, and D. Kamdar, “Recognizing creative leadership: Can creative idea expression negatively relate to perceptions of leadership potential?,” Journal of Experimental Social Psychology, vol. 47, no. 2, pp. 494–498, Mar. 2011**

The three items represent a unidimensional measure of leadership potential, tapping into overall leadership presence, authority/dominance dimension, and global effectiveness judgment. The scale also proved to be consistency reliabile, with Cronbach's alpha values of α = .86 is higher than the standard .70 (Nunnally, 1978). As the 3 points towards leadership potential. Adapting from the paper's procedure, we will evaluate all 3 points and average the score for subsequent analysis.

> Evaluators were tasked to rated on leadership potential using a 3-item scale: “How much leadership would this applicant exhibit?”, “How much control over the team's activities would this member exhibit?” and “I think the applicant is an effective leader.” (Study 2, p.496)

---

## 6. Expected Status

**Anderson, C., John, O. P., Keltner, D., & Kring, A. M. (2001). Who attains social status? Effects of personality and physical attractiveness in social groups. *Journal of Personality and Social Psychology, 81*(1), 116–132.**

> "We measured status with three separate peer ratings. Participants rated the other sorority members on the amount of status (1 = low status, 7 = high status) they had in the sorority, the amount of influence (1 = not influential, 7 = very influential) they had in the sorority, and their prominence in the sorority (1 = not visible, 7 = very visible)." (Study 2, p. 122)

> "Status was defined for the participants as 'the amount of prominence, respect, and influence' the individual held in the residence hall; the rating scale ranged from 1 (low) to 7 (high)." (Study 3, pp. 124–125)

Construct definition (p. 116): "Such face-to-face status is defined by the amount of respect, influence, and prominence each member enjoys in the eyes of the others."


---



---

## 8. Perceived Warmth

**Fiske, S. T., Cuddy, A. J. C., Glick, P., & Xu, J. (2002). A model of (often mixed) stereotype content: Competence and warmth respectively follow from perceived status and competition. *Journal of Personality and Social Psychology, 82*(6), 878–902.**

Item wording verified via a later review by one of the original authors: Fiske, S. T. (2018). Stereotype content: Warmth and competence endure. *Current Directions in Psychological Science, 27*(2), 67–73.

> "Warmth items include warm, trustworthy, friendly, honest, likable, and sincere (in order of priority)."


---

---

# 9. Mapping Table: Prompt Items to Original Source Items

Applies to both `prompt-design.md` and `prompt-design-region.md`. The evaluation criteria block is identical in both files; the two versions differ only in the `{{REGION}}` framing.

| JSON field | Prompt item (as used) | Original item (as published) | Source | Adaptation |
|---|---|---|---|---|
| `hireability` | "I would recommend extending a job offer to this candidate"; "Overall, I would evaluate this candidate positively." (1–7) | "I would recommend extending a job offer to this applicant"; "Overall, I would evaluate this applicant positively." (1–7) | Higgins & Judge (2004) | "applicant" → "candidate". Scale and item count unchanged. Rating from documents rather than post-interview. |
| `perceived_competence` | "This candidate seems competent, intelligent, skilled, and efficient." (1–5) | Competence items: competent, intelligent, skilled, efficient, assertive, confident | Fiske et al. (2002); items via Fiske (2018) | Trait adjectives → statement form. *assertive* and *confident* dropped (see Note A). Target changed from social group to individual candidate. 1–5 scale set by this study; not stated in Fiske (2018). |
| `person_job_fit` | "This candidate possesses the knowledge, skills, and abilities necessary to perform the duties of this specific job"; "This candidate can achieve a high level of performance in this particular job." (1–7) | "This applicant possesses the KSAs necessary to perform the duties of this specific job"; "I believe this applicant can achieve a high level of performance in this particular job." (1–7) | Higgins & Judge (2004) | "applicant" → "candidate"; "KSAs" expanded to full phrase; "I believe" dropped. Scale and item count unchanged. |
| `expected_performance` | "This candidate would adequately complete assigned duties"; "This candidate would meet the formal performance requirements of the job." (1–5) | IRB items 1 and 4: "Adequately completes assigned duties."; "Meets formal performance requirements of the job." | Williams & Anderson (1991), Table 1, p. 606 | 2 of 7 IRB items retained; both reverse-coded items omitted. Re-tensed from observed to predicted ("would"). 1–5 scale set by this study; response format not stated in the article (see Note B). |
| `leadership_potential` | "This candidate would exhibit a high degree of leadership"; "This candidate would exert control over the team's activities"; "I think this candidate is an effective leader." (1–7, 1 = strongly disagree, 7 = strongly agree) | 3-item scale: "How much leadership would this applicant exhibit?"; "How much control over the team's activities would this member exhibit?"; "I think the applicant is an effective leader." (α = .86) | Mueller, Goncalo, & Kamdar (2011), Study 2, p. 496 |Three items rated separately and averaged in analysis, following the source procedure. Items converted from interrogative to declarative form to fit agreement anchors; "applicant"/"member" → "candidate". Anchors inferred from Study 1, where all measures ran 1 (strongly disagree) to 7 (strongly agree); not stated for the Study 2 leadership items. The study's manipulation check (creativity, novelty, usefulness) and its competence and warmth control measures are not included. |
| `expected_status` | "the amount of status — prominence, respect, and influence — you would expect this candidate to hold in this role" (1 = low, 7 = high) | Status defined as "the amount of prominence, respect, and influence" the individual held; 1 (low) to 7 (high) | Anderson et al. (2001), Study 3, pp. 124–125 | Definition and scale retained verbatim. Dimension renamed from Status Fit to Expected Status: the source measures the level of status a person holds, not congruence between a person and a role. Shifted from status *observed* by peers over months to status *anticipated* in a role. Rater changed from group peer to evaluator working from documents. |
| `perceived_warmth` | "This candidate seems warm, trustworthy, friendly, and sincere." (1–5) | Warmth items: warm, trustworthy, friendly, honest, likable, sincere (in order of priority) | Fiske et al. (2002); items via Fiske (2018) | Trait adjectives → statement form. *honest* and *likable* dropped. Target changed from social group to individual candidate. 1–5 scale set by this study; not stated in Fiske (2018). |

## Notes on the three adaptations requiring justification

**Note A — Perceived Competence.** Fiske's competence item list includes *assertive* and *confident*, which belong to the agency dimension. Because this study manipulates agentic versus communal-relational framing in the cover letters, retaining those two adjectives would make the outcome partly a restatement of the independent variable. They were therefore dropped, leaving the four capability-focused adjectives. Perceived Competence and Perceived Warmth still overlap conceptually with the two framing conditions and are best read as manipulation checks rather than independent outcomes.

