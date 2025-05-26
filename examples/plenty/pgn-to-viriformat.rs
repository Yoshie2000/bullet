use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon::ThreadPoolBuilder;
use regex::Regex;
use std::env;
use std::io::{BufRead, BufReader, BufWriter};
use std::str::FromStr;
use std::{fs, process};
use viriformat::chess::board::movegen;
use viriformat::chess::board::{Board, GameOutcome};
use viriformat::chess::chessmove::Move;
use viriformat::chess::piece::PieceType;
use viriformat::chess::piece::{Colour, Piece};
use viriformat::chess::squareset::SquareSet;
use viriformat::chess::types::{CheckState, Square};
use viriformat::chess::CHESS960;
use viriformat::dataformat::Game;

pub fn san(board: &mut Board, m: Move) -> Option<String> {
    let check_char = match board.gives(m) {
        CheckState::None => "",
        CheckState::Check => "+",
        CheckState::Checkmate => "#",
    };
    if m.is_castle() {
        match () {
            () if m.to() > m.from() => return Some(format!("O-O{check_char}")),
            () if m.to() < m.from() => return Some(format!("O-O-O{check_char}")),
            () => unreachable!(),
        }
    }
    let to_sq = m.to();
    let moved_piece = board.piece_at(m.from())?;
    let is_capture =
        board.is_capture(m) || (moved_piece.piece_type() == PieceType::Pawn && Some(to_sq) == board.ep_sq());
    let piece_prefix = match moved_piece.piece_type() {
        PieceType::Pawn if !is_capture => "",
        PieceType::Pawn => &"abcdefgh"[m.from().file() as usize..=m.from().file() as usize],
        PieceType::Knight => "N",
        PieceType::Bishop => "B",
        PieceType::Rook => "R",
        PieceType::Queen => "Q",
        PieceType::King => "K",
    };
    let possible_ambiguous_attackers = if moved_piece.piece_type() == PieceType::Pawn {
        SquareSet::EMPTY
    } else {
        movegen::attacks_by_type(moved_piece.piece_type(), to_sq, board.pieces.occupied())
            & board.pieces.piece_bb(moved_piece)
    }
    .into_iter()
    .filter(|sq| {
        let m = Move::new(*sq, to_sq);
        let legal = if board.make_move_simple(m) {
            board.unmake_move_base();
            true
        } else {
            false
        };
        legal
    })
    .fold(SquareSet::EMPTY, |acc, sq| SquareSet::from_square(sq) | acc);

    let needs_disambiguation = possible_ambiguous_attackers.count() > 1 && moved_piece.piece_type() != PieceType::Pawn;
    let from_file = SquareSet::FILES[m.from().file()];
    let from_rank = SquareSet::RANKS[m.from().rank()];
    let can_be_disambiguated_by_file = (possible_ambiguous_attackers & from_file).count() == 1;
    let can_be_disambiguated_by_rank = (possible_ambiguous_attackers & from_rank).count() == 1;
    let needs_both = !can_be_disambiguated_by_file && !can_be_disambiguated_by_rank;
    let must_be_disambiguated_by_file = needs_both || can_be_disambiguated_by_file;
    let must_be_disambiguated_by_rank = needs_both || (can_be_disambiguated_by_rank && !can_be_disambiguated_by_file);

    let disambiguator1 = if needs_disambiguation && must_be_disambiguated_by_file {
        &"abcdefgh"[m.from().file() as usize..=m.from().file() as usize]
    } else {
        ""
    };
    let disambiguator2 = if needs_disambiguation && must_be_disambiguated_by_rank {
        &"12345678"[m.from().rank() as usize..=m.from().rank() as usize]
    } else {
        ""
    };
    let capture_sigil = if is_capture { "x" } else { "" };
    let promo_str = match m.promotion_type() {
        Some(PieceType::Knight) => "=N",
        Some(PieceType::Bishop) => "=B",
        Some(PieceType::Rook) => "=R",
        Some(PieceType::Queen) => "=Q",
        None => "",
        _ => unreachable!(),
    };
    let san = format!("{piece_prefix}{disambiguator1}{disambiguator2}{capture_sigil}{to_sq}{promo_str}{check_char}");
    Some(san)
}

// pub fn convert_castling(board: &mut Board, fen: &str) -> String {
//     let fen_chars = fen.as_bytes();
//     let split_idx =
//         fen_chars.iter().position(|&c| c == b' ').unwrap();
//     let (_, info_part) = fen_chars.split_at(split_idx);
//     let mut info_parts = info_part[1..].split(|&c| c == b' ');

//     let _ = info_parts.next();
//     let castling_part = info_parts.next();

//     let mut correct_castling_parts = Vec::<u8>::new();

//     match castling_part {
//         None => panic!(),
//         Some(b"-") => {},
//         Some(castling) if CHESS960.load(std::sync::atomic::Ordering::SeqCst) => {
//             for &c in castling {
//                 let flag = match c {
//                     b'K' => {
//                         let rook_square = (0..8).filter(|sq| {
//                             board.piece_array[*sq as usize].unwrap_or_else(|| Piece::WP).piece_type() == PieceType::Rook
//                                 && board.piece_array[*sq as usize].unwrap_or_else(|| Piece::WP).colour() == Colour::White
//                         }).max().map(|sq| Square::new(sq as u8).unwrap()).take().unwrap();
//                         b'H' - (7 - rook_square.inner())
//                     }
//                     b'Q' => {
//                         let rook_square = (0..8).filter(|sq| {
//                             board.piece_array[*sq as usize].unwrap_or_else(|| Piece::WP).piece_type() == PieceType::Rook
//                                 && board.piece_array[*sq as usize].unwrap_or_else(|| Piece::WP).colour() == Colour::White
//                         }).min().map(|sq| Square::new(sq as u8).unwrap()).take().unwrap();
//                         b'A' + rook_square.inner()
//                     },
//                     b'k' => {
//                         let rook_square = (56..64).filter(|sq| {
//                             board.piece_array[*sq as usize].unwrap_or_else(|| Piece::WP).piece_type() == PieceType::Rook
//                                 && board.piece_array[*sq as usize].unwrap_or_else(|| Piece::WP).colour() == Colour::White
//                         }).max().map(|sq| Square::new(sq as u8).unwrap()).take().unwrap();
//                         b'h' - (7 - rook_square.inner())
//                     },
//                     b'q' => {
//                         let rook_square = (56..64).filter(|sq| {
//                             board.piece_array[*sq as usize].unwrap_or_else(|| Piece::WP).piece_type() == PieceType::Rook
//                                 && board.piece_array[*sq as usize].unwrap_or_else(|| Piece::WP).colour() == Colour::White
//                         }).min().map(|sq| Square::new(sq as u8).unwrap()).take().unwrap();
//                         b'a' + rook_square.inner()
//                     },
//                     _ => panic!()
//                 };
//                 correct_castling_parts.push(flag);
//             }
//         }
//         Some(_shredder_castling) => {}
//     };

//     let updated_fen = match castling_part {
//         Some(_) => {
//             let mut parts: Vec<&str> = fen.splitn(3, ' ').collect();
//             parts[1] = if correct_castling_parts.is_empty() {
//                 "-"
//             } else {
//                 std::str::from_utf8(&correct_castling_parts).unwrap()
//             };
//             format!("{} {} {}", parts[0], parts[1], parts[2])
//         }
//         None => panic!(),
//     };
    
//     updated_fen
// }

fn convert_pgn(pgn_path: &str) {
    let input_file = fs::File::open(pgn_path).unwrap();
    let reader = BufReader::new(input_file);

    let output_file = fs::File::create(pgn_path.replace(".pgn", ".viri")).unwrap();
    let mut writer = BufWriter::new(output_file);

    let mut new_line_found = false;
    let mut result = GameOutcome::Ongoing;
    let mut start_fen = String::new();

    let token_regex = Regex::new(r"(\{[^}]*\}|\S+)").unwrap();

    for line in reader.lines() {
        let line = line.unwrap();

        if line.starts_with("[Result") {
            let rstr = line[9..line.rfind('"').unwrap()].to_string();
            result = match rstr.as_str() {
                "1-0" => GameOutcome::WhiteWin(viriformat::chess::board::WinType::Adjudication),
                "0-1" => GameOutcome::BlackWin(viriformat::chess::board::WinType::Adjudication),
                _ => GameOutcome::Draw(viriformat::chess::board::DrawType::Adjudication),
            };
        }

        if line.starts_with("[FEN") {
            start_fen = line[6..line.rfind('"').unwrap()].to_string();
        }

        if line.starts_with("[Variant \"fischerandom\"]") {
            CHESS960.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        if new_line_found && !line.is_empty() {
            // Init board
            let mut start_board = Board::new();
            let _ = start_board.set_from_fen(&start_fen.as_str()).unwrap();
            start_board.set_fullmove_clock(8); // this is probably somewhat the average of UHO_Lichess + random moves (OB used to not have the correct fullmove numbers in the PGN start fen)
            let mut game = Game::new(&start_board);
            game.set_outcome(result);

            let mut current_board = Board::new();
            let _ = current_board.set_from_fen(&start_fen.as_str()).unwrap();

            let mut is_move = true;
            let mut current_move: Option<Move> = None;
            let mut current_eval: i16;

            for token_match in token_regex.find_iter(&line) {
                let token = token_match.as_str().to_string();

                if is_move {
                    // Parse the move, unless the game is over
                    if token == "1-0" || token == "0-1" || token == "1/2-1/2" {
                        break;
                    }

                    let replaced_token = token.replace("+", "").replace("+", "").replace("#", "").replace("x", "");
                    current_move = if !token.contains("-") && !token.contains("=") && replaced_token.len() == 5 {
                        // Complete move info is given
                        println!("{token} {replaced_token}");
                        let from = Square::from_str(&replaced_token[1..=2]).unwrap();
                        let to = Square::from_str(&replaced_token[3..=4]).unwrap();
                        Some(Move::new(from, to))
                    } else {
                        let first_matching_move = current_board
                            .legal_moves()
                            .iter()
                            .find(|m| san(&mut current_board, **m).unwrap() == token)
                            .take()
                            .cloned();
                        if first_matching_move == None {
                            println!("Token: {token}");
                            current_board
                                .legal_moves()
                                .iter()
                                .for_each(|m| print!("{} ", san(&mut current_board, *m).unwrap()));
                            println!("");
                            current_board.piece_array.iter().for_each(|p| {
                                print!("{} ", if let Some(p2) = p { p2.to_string() } else { "X".to_string() });
                            });
                            println!("");
                        }
                        Some(first_matching_move.unwrap().clone())
                    };
                } else {
                    // Parse the evaluation
                    let mut eval_str = token.split_whitespace().next().unwrap_or("").to_string();
                    if !eval_str.contains('M') {
                        eval_str.retain(|c| c != '.' && c != '{');
                        current_eval = eval_str.parse().unwrap_or(0);
                        if current_board.turn() == Colour::Black {
                            current_eval = -current_eval;
                        }
                    } else {
                        // Ignore mate scores
                        break;
                    }

                    // Add the move
                    game.add_move(current_move.unwrap(), current_eval);
                    let result = current_board.make_move_simple(current_move.unwrap());
                    if !result {
                        panic!("Makemove was illegal");
                    }
                }

                is_move = !is_move;
            }

            if game.len() >= 10 && game.len() <= 250 {
                let _ = game.serialise_into(&mut writer).unwrap();
            }
        }

        if line.is_empty() {
            new_line_found = !new_line_found;
        }
    }
}

fn main() {
    let pgn_folder = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: <program> <pgn_folder>");
        process::exit(1);
    });

    let pgn_paths: Vec<String> = fs::read_dir(pgn_folder)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|e| e.is_file()))
        .map(|entry| entry.path().display().to_string())
        .collect::<Vec<_>>();

    ThreadPoolBuilder::new().num_threads(20).build_global().unwrap();

    pgn_paths.par_iter().for_each(|pgn_path| {
        convert_pgn(pgn_path);
    });
}
