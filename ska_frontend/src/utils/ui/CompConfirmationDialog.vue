<!-- eslint-disable vue/no-mutating-props -->
<script setup lang="ts">
import { onMounted, ref, watch } from 'vue';

// props
const props = defineProps<{
    open: boolean,
    title: string,
}>();

// outputs
const emit = defineEmits<{
    (e: 'action', isConfirm: boolean): void,
}>();


// hooks
const isLoading = ref(false);
const confirmationDialog = ref<HTMLDialogElement | null>(null);

// lifecycles
onMounted(() => {
    if (props.open) {
        isLoading.value = false;
        showModal();
    }
});

watch(props, (newProps) => {
    if (newProps.open) {
        isLoading.value = false;
        showModal();
    } else {
        close();
    }
});

// functions

function showModal(): void {
    confirmationDialog.value?.showModal();
}

function close(): void {
    confirmationDialog.value?.close();
}

function onCancel(): void {
    isLoading.value = true;
    emit('action', false);
}

function onConfirm(): void {
    isLoading.value = true;
    emit('action', true);

}


defineExpose({
    showModal,
    close
});

</script>

<template>
    <dialog ref="confirmationDialog" id="confirmationDialog" class="modal">
        <div class="modal-box">
            <h3 class="font-bold text-lg">
                {{ props.title }}
            </h3>
            <slot></slot>
            <form @submit.prevent="onConfirm">
                <div class="modal-action">
                    <form method="dialog">
                        <button class="btn" @click.prevent="onCancel" :disabled="isLoading">Cancel</button>
                    </form>
                    <button class="btn btn-primary" type="submit" :disabled="isLoading">
                        Confirm
                    </button>
                </div>
            </form>
        </div>
    </dialog>
</template>