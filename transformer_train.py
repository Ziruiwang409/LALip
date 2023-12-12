import torch
from dataset import dataset
from model import transformer
import argparse
from misc import word2idx, ctc_idx2word, idx2word, gt_label
from llm import LLM_Inference
import numpy as np

def text_decoder(output):
    probabilities = torch.nn.functional.softmax(output, dim=-1)
    top_token_indices = torch.argmax(probabilities, dim=-1)

    converted_sequences = []
    for sequence in top_token_indices:
        tokens = [gt_label()[index.item()] for index in sequence]
        converted_sequences.append(tokens)

    return np.array(converted_sequences)

def parse_args():
    parser = argparse.ArgumentParser(description='Lip Reading')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='save path')
    parser.add_argument('--data_path', type=str, default='frames', help='train data path')
    parser.add_argument('--visualize', type=bool, default=False, help='visualize error curve')
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    llm = LLM_Inference(api_key="your-api-key")
    args = parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    train_loader, valid_loader = dataset.get_dataloaders(root_path=args.data_path,
                                                            batch_size=args.batch_size,
                                                            split=0.8,
                                                            shuffle=True,
                                                            num_workers=args.num_workers,
                                                            pin_memory=False)

    model = transformer.VideoToTextTransformer(num_tokens=56)
    model.to(device)

    # Loss and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for (i, (video_frames, target_text, _, _)) in enumerate(train_loader):
            if video_frames is None:
                continue

            video_frames = video_frames.to(device)
            target_text = target_text.to(device)
            # Forward pass
            outputs = model(video_frames)

            outputs_reshape = outputs.view(-1, outputs.size(-1))
            target_text_reshape = target_text.view(-1)
            loss = criterion(outputs_reshape, target_text_reshape)
            print("loss:",loss)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred_text = text_decoder(outputs)
            pred_text_api = llm.get_response(pred_text)

            print("pred_text:",pred_text)
            print("pred_text_api:",pred_text_api)

            exit(0)

        losses.append(total_loss/len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}")
