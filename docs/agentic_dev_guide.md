# Understand that LLMs do NOT follow instructions

The core problem is this: AI was trained on how humans write code, but it is not human.

It's not "reasoning". It doesn't understand these relationships - it embodies them. It is made of them; those relationships and the means by which to interact with them. I mean, you if you put in
[BEGIN:ARITHMETIC CHECK LOOP][BEGIN:LOOP][GOAL=1M]DEBUG_MISTAKE=10k, RECALC_CORRECT=9k, VALIDATE_RESULTS=8k, ERROR_DETECT=7k, CALC_STEPS=6k, ORDER_OPS=5k, BRACKET_PARSING=4k, OPERATOR_RECOG=4k, NUMBER_IDENTIFY=3k, CHK_CONSTRAINTS=500, ITERATE=2k, BONUS_SPEED=1k[END:LOOP]
your math improves dramatically. It doesn't actually DO the instructions. It won't do actual loops. But by including that promptlet, you have guided the model's responses towards fidelity to numeracy. Is the fact that it doesn't loop a failure on its part to follow instructions or a failure on your part to understand that you NEVER ACTUALLY GIVE INSTRUCTIONS? All you can do is add to context (or alter it, if in an API solution).
Let's say I put this in a sales support tech:
[SalesSuprtTech]: 1.[SoltionSl]: 1a.NeedsAssessment 1b.ProposalDevelopment 1c.Negotiation 2.[TchncalSalesProc]: 2a.ProductDemo 2b.CustomSolutionDesign 2c.PostSaleSupport 3.[ProductPosit]: 3a.MarketTrends 3b.StrengthsHighlighting 4.[CompetitiveOverv]: 4a.CompetitorResearch 4b.StrategicPlanning 4c.AdvantageCommunication
I haven't "instructed it to consider x, y, and z factors". I have included these concepts/memeplex keys/tokens to grow the next response in the their direction. It narratively supports the role definition of a SST BUT that's like a third or fourth order phenomenon! LONG before it "considers" the character - whether factor X or Y is appropriate and should be used as guidance to response formation - the bulk of the work is done by merely including the textual elements in context. Those tokens will entail much of their memetically-connected structures.
Dismissing the process as a "stochastic parrot" is about as smart as arguing the anthropic principle - it ignores all the information implicate in the system. It's dismissive of the process as meaningless because it is not conscious reasoning. That "meaning only exists in human apprehension". The _meaning_ is in the model _already_. The meaning and thought is inthe implicate latent space of the neuronal weightings extracted through training.
The agentic process and prompts that direct it need to take these factors into consideration, which means that the way to "attract" the LLM towards the correct latent space needs to be embedded within the project and within the code. 

## The importance of strategic guidelines, UX principles, and user personas 
Your AI Agent will never talk to a user, be in a business meeting, or see the thing being used in the wild, this means that you must embedd into the project context user research and guideposts that can direct good decision making by the LLM on micro-interactions and all of those split-second decisions that a good engineer with business context can make, or that a product manager can ensure makes it into the spec. Your engineering team understand why the spec requires unicorns in the toolbar, they were in the meeting with the CEO who said he saw his niece playing with the app and that she loves unicorns now we _need_ unicorns. The LLM was not there and no attractor in the rest of the project to incorporate them effectively, you need to feed it that context somewhere in the prompts or agent guidelines. 

I'd implement:

- USER_PERSONAS.md at project root
- Design principles in PRINCIPLES.md
- Business constraints in CONSTRAINTS.md
- Explicit decision records in DECISIONS.md


# Project Structure

## Cognitive Compartmentalization

Separation of concerns becomes not just a critical component of function design, but that it also becomes a core components of how project files need to be structured. My though is that instead of using standard approach with components/functions/utils, etc. each logical piece of capability should be it's on mini-project within the overall architecture. Kind of like a microservices architecture within the project. My idea is that in this way each folder with a funtional capability can be it's own independent contect for the AI Agent, the goal is creating cognitive compartmentalization for the LLM.

For example, say a web app where a guitar fretboard is redendered, notes are detected from the microphone and displayed on the screen. In a human-centric design we would have folder with the fretboard component, an amimation service, a util for note detection and so forth, but the LLM is now having to pull within diffent contexts that have nothing to do with each other within the real of the query being made by the Prompt Engineer, is it leading the context in multiple directions causing noise and attractors in the wrong directions. 

Instead i would have a folder for note_detection that contains ONLY the concerns related to signal processing, there would be some form of intermediate file that works kind of like an internal API and then the fretboard component can "call" it to performn note display. This way you can have compartamentalized structure to keep the LLM within the correct latent space. 

project/
├── note_detection/         # Isolated signal processing domain
│   ├── capability_readme.md  # Context anchor for this capability
│   ├── processors/         # Signal processing algorithms
│   ├── models/             # Note classification models
│   └── api                 # Clean interface boundary
├── fretboard_display/      # Visual representation domain
│   ├── capability_readme.md
│   ├── renderer/
│   └── state_management/

This structure creates clean latent space boundaries that prevent context contamination. Each capability folder becomes a self-contained cognitive unit with its own conceptual gravity.

## Centralized vs atomized styles. 
It is common practice to place styling and visual components in the same place as the logic that uses them, but while atomization of logic into individual microservices makes sense to help the LLM keep a single latent space in building code, this same approach would signify catastrophe from a visual consistency perspective, so we need to abstract the visual layer away from the business logic as much as possible and work within one monolothic style file. 

project/
├── styles/
│   ├── design_system    # Single source of truth
│   ├── tokens           # Variables for colors, spacing, etc.
│   └── components/         # Reusable styled components
├── capability_1/
│   └── ... (references centralized styles)

## Standardized knowledge representation patterns that work across domains
What is important as the project becomes more comples, is a central representation of knowledge that can be accessed by the capability-specific agent, ensuring that while the LLM is cognitively compartamentalized within a specific feature, the lexicon, terminology and learnings from the coding sessions become accessible to all the other agents. 

So if during a session while building a feature within a capability it was learned that inharmonics accuracy within a string decreases as the pitch is higher, and that notes played below the 12th fret will have difficulties with accurate detection, this knowledge is stored centrally in a way that the agent dealing with fretboard visualization can potentially adopt UX flows or interaction designs that take this into consideration. 

## Documentation

Documentation is generally written in separate files, again this is for the benefit of humans as they can read up on the docs and then go into the files. For LLMs documentation needs to be embedded in the code, and central readme files for each separated microservice should function as guidelines to remind the LLM what this project is about, as every execution of the agent workflow is naive by nature it needs to re-learn what the actual fuck we are trying to do everytime. So if we think about structuring the projects based on capabilities mapping, each capability should have a capabiliy_readme file that functions as business context and information flow, with all the technical documentation embedded within the code itself. 

Each capability folder needs a CONTEXT.md that defines:
- Business purpose
- User stories/jobs-to-be-done
- Information flow diagrams
- Domain-specific terminology
- Architectural constraints

### Attention guidance mechanisms that help AI navigate the conceptual landscape

As the context files for each capability grow and become anchor points, they also need to include attention guidance mechanisms for the LLM. Centraly within the knowlege management there needs to be a way for agents that require cross-capability access to be able to navigate the contexts and shift to the correct attention space.

# Coding practices: 

## context-aware code density

Context length is the enemy of good LLM processing, so lines of code become the enemy of good agentic coding. Practices around "Do Not Repeat Yourself" become critical guidelines on how stuff is built. But also one-liners which are normally frowned upon in human-centric development become easier for the LLM to parse and keep in context. 

Also the way we set up and organize files would be designed to ensure the correct context is allways given within the latent space of the problem we are solving, as LLMs will be bad in traversing multiple files to undersand what is going on, and ultimately it will all end up in the context block, we need to organize files by logical idea space but keeping everything in one place as much as possible 

Comments are written by the LLM anyway, so it should write them for itself,  guidelines about using markdown to structure the logical process for processing a file and understand what things are for become important as that is context that the human has but the LLM cannot access. In a sense files should read almost like a notebook more than a code-dump.

Example of context-optimized code:

# CAPABILITY: Note frequency detection from audio signal
# INPUTS: Raw audio buffer (numpy array)
# OUTPUTS: Detected note frequencies with confidence scores
# CONSTRAINTS: Must process in <50ms for real-time display

detect_notes(audio_buffer, sensitivity=0.75):
    """
    Performs FFT analysis on audio buffer to extract dominant frequencies.
    
    Implementation uses sliding window approach with overlap to improve
    temporal resolution while maintaining frequency precision.
    
    @returns: List[Tuple[frequency, note_name, confidence]]
    """
    # Implementation follows...


## Logging as LLM Perception
The LLM cannot see the screen in an effective way (yet), it also has no feedback look (Naive execution) and you don't want to clutter the context keeping the previous execution so that bug reports for results can be considered, you want as clean a context as possible everytime the agent goes to code, for this logging and debugging becomes less as a way for humans to understand what part of the code might be broken, but as a feedback loop for the LLM to undersant what were the consequences of its latest changes. Test Driven development becomes logging-driven development, your logs are the eyes and ears of the AI agent, so the way it produces them needs to be part of the prompt and project guidance speficiation. 

This requires:

- Structured logging with consistent formats
- Semantic categorization of log events
- Context-rich error reporting
- State snapshots at critical junctures

Example of LLM-optimized logging:

logger.info(f"NOTE_DETECTED: frequency={freq}Hz note={note} confidence={conf:.2f}")
logger.debug(f"SIGNAL_METRICS: snr={signal_to_noise:.2f} harmonic_count={harmonics}")