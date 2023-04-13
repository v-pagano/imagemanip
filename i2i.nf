nextflow.enable.dsl=2

include { publishResults } from '../nextflow-modules/publish'

process t2i {

    input:
        val textPrompt
        val numImages
        val steps
        val guidance
        val seed

    output:
        path "*.png"

    time '1h'
    clusterOptions '--gres=gpu:A100:1'
    container 'oras://ghcr.io/v-pagano/imagemanip:latest'

    script:
    def p = numImages != '' ? " --nImages ${numImages} " : '' 
    if (steps != '') {
        p = "${p} --steps ${steps} "
    } 
    if (guidance != '') {
        p = p + " --guidance ${guidance} "
    }
    if (seed != '') {
        p = p + " --seed ${seed} "
    }
    println(p)
    """
        t2i.py --prompt '${textPrompt}' ${p}
    """

}

params.nImages = 5
params.steps = 50
params.guidance = 7.5
params.seed = 0

workflow {
    t2i(params.prompt, params.nImages, params.steps, params.guidance, params.seed)
    publishResults(t2i.out, params.outputFolder)
}