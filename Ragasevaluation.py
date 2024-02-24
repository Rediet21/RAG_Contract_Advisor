from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.metrics import faithfulness , answer_relevancy, context_precision , context_recall

from Front_end.front import handle_input,get_pdf_text,get_text_chunks,get_embedding,get_conversation


faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
context_rel_chain = RagasEvaluatorChain(metric=context_precision)
context_recall_chain = RagasEvaluatorChain(metric=context_recall)
from PyPDF2 import PdfReader

#Bring the document
file = PdfReader('data/Raptor Contract.pdf')
def get_pdf_text(file):
    
    text = " "
    for page in file.pages:
        text += page.extract_text()
    return text
#import functions
raw_text = get_pdf_text(file)
text_chunks = get_text_chunks(raw_text)
vectordb = get_embedding(text_chunks)
conversation = get_conversation(vectordb)


eval_questions = [
    "Under what circumstances and to what extent the Sellers are responsible for a breach of representations and warranties?"
    "Would the Sellers be responsible if after the closing it is determined that there were inaccuracies in the representation provided by them where such inaccuracies are the resolute of the Sellers’ gross negligence?"
    "How much is the escrow amount?"
    "What is the purpose of the escrow?"
    "May the Escrow Amount serve as a recourse for the Buyer in case of breach of representations by the Company?"
    "Are there any conditions to the closing?"
    "Are Change of Control Payments considered a Seller Transaction Expense?"
    "Yes. (See defining of Sellers Transaction Expenses)."
    "Would the aggregate amount payable by the Buyer to the Sellers be affected if it is"
    "Does the Buyer need to pay the Employees Closing Bonus Amount directly to the Company’s employees?"
    "Does any of the Sellers provide a representation with respect to any Tax matters related to the Company?"
    "Is any of the Sellers bound by a non-competition covenant after the Closing?"
    "Whose consent is required for the assignment of the Agreement by the Buyer?"
    "Does the Buyer needs the Sellers’ consent in the event of an assignment of the Agreement to a third party who is not a Buyer’s Affiliates?"
]


eval_answers = [
    "Cloud Investments Ltd. (“Company”) and Jack Robinson (“Advisor”)"
    "According to section 4:14 days for convenience by both parties. The Company may terminate without notice if the Advisor refuses or cannot perform the Services or is in breach of any provision of this Agreement."
    "According to section 6: 1. Fees of $9 per hour up to a monthly limit of $1,500, 2. Workspace expense of $100 per month, 3. Other reasonable and actual expenses if approved by the company in writing and in advance"
    "Under section 1.1 the Advisor can’t assign any of his obligations without the prior written consent of the Company, 2. Under section 9  the Advisor may not assign the Agreement and the Company may assign it, 3 Under section 9 of the Undertaking the Company may assign the Undertaking."
    "According to section 4 of the Undertaking (Appendix A), Any Work Product, upon creation, shall be fully and exclusively owned by the Company."
    "Yes. During the term of engagement with the Company and for a period of 12 months thereafter."
    "No. See Section 6.1, Billable Hour doesn’t include meals or travel time. "
    "Rabin st, Tel Aviv, Israel"
    "No. According to section 8 of the Agreement, the Advisor is an independent consultant and shall not be entitled to any overtime pay, insurance, paid vacation, severance payments or similar fringe or employment benefits from the Company."
    "If the Advisor is determined to be an employee of the Company by a governmental authority, payments to the Advisor will be retroactively reduced so that 60% constitutes salary payments and 40% constitutes payment for statutory rights and benefits. The Company may offset any amounts due to the Advisor from any amounts payable under the Agreement. The Advisor must indemnify the Company for any losses or expenses incurred if an employer/employee relationship is determined to exist." 
]


examples = [
    {"query": q, "ground_truth": [eval_answers[i]]}
    for i, q in enumerate(eval_questions)
]


# results = conversation({"query": eval_questions[1]})
# results["result"]

from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.metrics import(
     faithfulness,
     answer_relevancy,
     context_precision,
     context_recall,
)
faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
context_rel_chain = RagasEvaluatorChain(metric=context_precision)
context_recall_chain = RagasEvaluatorChain(metric=context_recall)

predictions = conversation.batch(examples)

print("evaluating...")
r = faithfulness_chain.evaluate(examples, predictions)
r

# evaluate context recall
print("evaluating...")
r = context_recall_chain.evaluate(examples, predictions)
# data = {
#     "question": eval_questions,
#     "answer": answers,
#     #"contexts": contexts,
#     "ground_truths": eval_answers
# }



# handle_input
# response = conversation({'question': query})
