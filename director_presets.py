// SPD (Style Preset Database)  
const DirectingDNA = {  
    spikeLee: {  
        coverage: [  
            { shot: "doubles_2s_dutch", freq: 0.4 },  
            { shot: "dolly_push_tight", freq: 0.3 }  
        ],  
        sound: {  
            scoreCut: "syncopated_jazz_break",  
            dialogueMix: { reverbDecay: 2.1 }  // More echo for that stage-y feel  
        },  
        momentEmphasis: (turn) => turn ? "release TO conflict" : "constrict"  
    },  
    tarantino: {  
        coverage: [  
            { shot: "trunk_shot", freq: 0.1 },  // Rare but iconic  
            { shot: "two_shot_low", freq: 0.6 }  
        ],  
        sound: {  
            scoreCut: "surf_rock_envelope",  
            dialogueMix: { reverbDecay: 0.3 }  // Dry + intense proximity  
        },  
        momentEmphasis: (turn) => "comic_beat_post_violence"  
    }  
};