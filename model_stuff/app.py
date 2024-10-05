import gradio as gr
from ps4_eval.eval import sample_new_sequence
from ps4_data.get_embeddings import generate_embedings


def pred(residue_seq):
    embs = generate_embedings(residue_seq)["residue_embs"]["0"]
    preds = sample_new_sequence(embs, "ps4_models/Mega/PS4-Mega_loss-0.633_acc-78.176.pt")
    return preds


iface = gr.Interface(
    fn=pred,
    title="Protein Secondary Structure Prediction with PS4-Mega",
    description="🧬 Predict protein secondary structure from single sequence input using PS4-Mega and ProtT5-XL-UniRef50 (Elnaggar et al., 2020).\n💻 Official github repo: https://github.com/omarperacha/ps4-dataset\n📝 Offical paper: https://www.biorxiv.org/content/10.1101/2023.02.28.530456",
    inputs=[gr.Textbox(label="Residue Sequence", value="")],
    outputs=[gr.Textbox(label="Secondary Structure", value="")],
    examples=[
        ["HXHVWPVQDAKARFSEFLDACITEGPQIVSRRGAEEAVLVPIGEWRRLQAAA"],
        ["AHKLFIGGLPNYLNDDQVKELLTSFGPLKAFNLVKDSATGLSKGYAFCEYVDINVTDQAIAGLNGMQLGDKKLLVQRASVGAKNA"]
    ]
)
iface.queue().launch(debug=True)
