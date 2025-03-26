use crate::embeddings::{
    input_embeddings::InputEmbeddings,
    pos_embeddings::{self, PosEmbeddings},
};
use candle_core::Device;
use candle_nn::Dropout;
use tokenizers::{Result, Tokenizer};

mod embeddings;

const D_MODEL: usize = 512;

fn main() -> Result<()> {
    // load a pre-trained tokenizer
    let tokenizer = Tokenizer::from_file("./src/tokenizer/wordlevel-wiki.json")?;

    let encoding = tokenizer.encode(("Welcome to the library. ", "test this out"), true)?;
    println!("tok: {:?}", encoding.get_tokens());
    // tok: ["welcome", "to", "the", "library", ".", "test", "this", "out"]
    println!("ids: {:?}\n", encoding.get_ids());
    // ids: [5807, 11, 5, 1509, 7, 681, 48, 92]

    let vocab_size = tokenizer.get_vocab_size(true);
    let token_ids = encoding.get_ids();

    let device = Device::Cpu;
    let input_embeds = InputEmbeddings::new(vocab_size, D_MODEL, &device)?;
    let embeddings = input_embeds.forward(&token_ids, &device)?;
    println!("vector embeddings:\n {}", embeddings);

    let mut pe = PosEmbeddings::new(8, D_MODEL, Dropout::new(0.3), &device)?;
    println!("pos_embeddings main:\n {}", pe.pos_embeddings);

    let encoder_input = pe.forward(embeddings)?;
    println!("encoder_input: {}", encoder_input);

    Ok(())
}
