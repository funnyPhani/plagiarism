#![warn(rust_2018_idioms)]
use std::io::Write;
use std::sync::Mutex;

use actix_web::{middleware, web, App, HttpResponse, HttpServer};
use chrono::Local;
use env_logger::Builder;
use log::{info, LevelFilter};
use serde::{Deserialize, Serialize};

use models::models::Model;

mod models;

#[derive(Debug, Serialize, Deserialize)]
struct IndexRequest {
    text_a: String,
    text_b: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexResponse {
    plagiarism: bool,
    accuracy: f32,
}

async fn index(item: web::Json<IndexRequest>, model: web::Data<Mutex<Model>>) -> HttpResponse {
    let model = model.lock().unwrap();
    let text_a = vec![item.text_a.to_owned()];
    let text_b = vec![item.text_b.to_owned()];
    let prediction = model.check_plagiarism(text_a, text_b).unwrap();
    HttpResponse::Ok().json(prediction)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "actix_web=info");

    Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S:%f"),
                record.level(),
                record.args()
            )
        })
        .filter(None, LevelFilter::Info)
        .init();

    let model_name = "plagiarism multilang detection version 0.1";
    let model = Model::new(
        model_name.to_owned(),
        "plagiarism_binary_multilang_classification1_model.h5".to_owned(),
    );
    // Create some global state prior to building the server
    #[allow(clippy::mutex_atomic)] // it's intentional.
    let model = web::Data::new(Mutex::new(model));

    info!("Loaded: {}", model_name);

    HttpServer::new(move || {
        App::new()
            .app_data(model.clone()) // add shared state
            .wrap(middleware::Logger::default())
            .data(web::JsonConfig::default().limit(8192))
            .service(web::resource("/").route(web::post().to(index)))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
