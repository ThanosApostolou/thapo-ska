<!-- eslint-disable vue/no-mutating-props -->
<script setup lang="ts">
import { ref } from 'vue';
import { ChatPacketType, DtoChatPacket } from './dtos/dto_chat_packet';
import { AskAssistantQuestionRequest } from './dtos/dto_ask_assistant_question';
import { ServiceAssistant } from './service_assistant';
import { DtoAssistantOptions, type DtoLlmData } from './dtos/dto_fetch_assistant_options';
import { DtoChatDetails } from './dtos/dto_chat_details';
import { useGlobalStore } from '@/domain/global_state/global_store';
import { DtoErrorPacket } from '@/utils/error/errors';

// props
const props = defineProps<{
    chatDetails: DtoChatDetails | null,
    assistantOptions: DtoAssistantOptions | null,
}>();

// outputs
const emit = defineEmits<{
    (e: 'success', chat_id: number): void
}>();


// hooks
const globalStore = useGlobalStore();
const isLoading = ref<boolean>(false);

const chatDialog = ref<HTMLDialogElement | null>(null);
const selectedLlm = ref<DtoLlmData | null>(props.assistantOptions?.llms[0] || null);
const chat_name = ref<string>('');
const prompt = ref<string>('');
const temperature = ref<number>(0);
const top_p = ref<number>(0);
const isEditPrompt = ref<boolean>(false);
const errorPackets = ref<DtoErrorPacket[]>([]);

// functions
function showModal(): void {
    chatDialog.value?.showModal();
}

function close(): void {
    chatDialog.value?.close();
}



function onSelectChange() {
    isEditPrompt.value = false;
    prompt.value = selectedLlm.value?.default_prompt || '';
    temperature.value = 0;
    top_p.value = 0;

}

async function onSubmit() {
    const chatDetails = new DtoChatDetails({
        chat_id: null,
        user_id: globalStore.globalStore.userDetails?.user_id || -1,
        chat_name: chat_name.value,
        llm_model: selectedLlm.value?.name || '',
        prompt_template: isEditPrompt.value ? prompt.value : null,
        temperature: isEditPrompt.value ? temperature.value : null,
        top_p: isEditPrompt.value ? top_p.value : null,
        default_prompt: selectedLlm.value?.default_prompt || '',
    });

    isLoading.value = true;
    errorPackets.value = [];
    const result = await ServiceAssistant.createChat(chatDetails);
    if (result.isOk()) {
        const data = result.data;
        isLoading.value = false;
        emit('success', data.chat_id);
    } else {
        const error = result.error;
        console.log('error', error)
        errorPackets.value = error.packets;
        isLoading.value = false;
    }
}


defineExpose({
    showModal,
    close
});

</script>

<template>
    <dialog ref="chatDialog" id="chatDialog" class="modal">
        <div class="modal-box">
            <h3 class="font-bold text-lg">Edit Chat
                <template v-if="props.chatDetails == null">Add Chat</template>
                <template v-else>Edit Chat</template>
            </h3>
            <form v-if="props?.assistantOptions != null" @submit.prevent="onSubmit">
                <div>
                    <label for="llms" class="form-control">Choose a LLM:</label>
                    <select v-model="selectedLlm" name="llms" id="llms" class="select select-bordered"
                        @change="onSelectChange()">
                        <option v-for="llm of props.assistantOptions.llms" :key="llm.name" :value="llm">
                            {{ llm.name }}
                        </option>
                    </select>

                    <div class="form-control">
                        <label class="label cursor-pointer">
                            <span class="label-text">Chat Name</span>
                            <input type="text" placeholder="Chat Name" v-model="chat_name"
                                class="input input-bordered" />
                        </label>
                    </div>

                    <div class="form-control">
                        <label class="label cursor-pointer">
                            <span class="label-text">Edit prompt</span>
                            <input type="checkbox" v-model="isEditPrompt" class="checkbox" />
                        </label>
                    </div>

                    <div class="form-control">
                        <label class="label cursor-pointer">
                            <span class="label-text">Temperature</span>
                            <input type="number" step="0.01" min="0" max="1" placeholder="Temperature"
                                v-model="temperature" class="input input-bordered" :disabled="!isEditPrompt" />
                        </label>
                    </div>

                    <div class="form-control">
                        <label class="label cursor-pointer">
                            <span class="label-text">Top-P</span>
                            <input type="number" step="0.01" min="0" max="1" placeholder="Top-P" v-model="top_p"
                                class="input input-bordered" :disabled="!isEditPrompt" />
                        </label>
                    </div>

                    <div class="form-control">
                        <label class="label cursor-pointer">
                            <span class="label-text">Prompt</span>
                            <textarea v-model="prompt" placeholder="Prompt" :disabled="!isEditPrompt"
                                class="textarea textarea-bordered textarea-xs w-full max-w-xs" rows="16">
                </textarea>
                        </label>
                    </div>

                </div>

                <div v-if="errorPackets.length > 0" role="alert" class="alert alert-error">
                    <p v-for="(errorPacket, index) in errorPackets" :key="index">{{ errorPacket.message }}</p>
                </div>

                <div class="modal-action">
                    <form method="dialog">
                        <!-- if there is a button in form, it will close the modal -->
                        <button class="btn" :disabled="isLoading">Cancel</button>
                    </form>
                    <button class="btn" type="submit" :disabled="isLoading">
                        <span v-if="isLoading" class="loading loading-spinner loading-sm"></span>Add
                    </button>
                </div>
            </form>
        </div>
    </dialog>
</template>