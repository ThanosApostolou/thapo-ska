use leptos::{create_rw_signal, expect_context, RwSignal};
use openidconnect::{core::*, *};

use crate::modules::auth::DtoUserDetails;

#[derive(Clone, Debug)]
pub struct GlobalStore {
    pub user_details: RwSignal<Option<DtoUserDetails>>,
    pub refresh_token: RwSignal<Option<RefreshToken>>,
    pub id_token: RwSignal<
        Option<
            IdToken<
                EmptyAdditionalClaims,
                CoreGenderClaim,
                CoreJweContentEncryptionAlgorithm,
                CoreJwsSigningAlgorithm,
                CoreJsonWebKeyType,
            >,
        >,
    >,
    pub access_token: RwSignal<Option<AccessToken>>,
}

impl GlobalStore {
    pub fn initialize_default() -> GlobalStore {
        let user_details: RwSignal<Option<DtoUserDetails>> = create_rw_signal(Option::None);
        let refresh_token: RwSignal<Option<RefreshToken>> = create_rw_signal(Option::None);
        let id_token: RwSignal<
            Option<
                IdToken<
                    EmptyAdditionalClaims,
                    CoreGenderClaim,
                    CoreJweContentEncryptionAlgorithm,
                    CoreJwsSigningAlgorithm,
                    CoreJsonWebKeyType,
                >,
            >,
        > = create_rw_signal(Option::None);
        let access_token: RwSignal<Option<AccessToken>> = create_rw_signal(Option::None);
        GlobalStore {
            user_details,
            refresh_token,
            id_token,
            access_token,
        }
    }

    pub fn expect_context() -> RwSignal<GlobalStore> {
        expect_context::<RwSignal<GlobalStore>>()
    }
}
